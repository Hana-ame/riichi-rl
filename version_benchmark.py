"""版本基准赛：每个版本都在**同一张强桌**上打，记录胜率/场均得分，和最新的
强版本（teacher47_v3）直接对比。

用户要求（2026-11-05）：「都记录一下各个版本胜率场均得分，必须和新的厉害的比。」

设计：
- 参照桌固定为「新强版本 + 人类牌谱教师 bc_v1 + 强规则基线 greedy_tenpai」
  （`--refs` 可改；规则名从 `policies.POLICIES` 识别，其余按 ckpt 载入并自动
  识别架构）；
- 每个待测版本轮流坐 4 个座位（座位轮换，消除庄家偏差），每版打 N 局；
- 记录每版：和牌率、场均得分（本局净点数）、放铳率；同时记录参照系自身读数；
- 同一参照桌下所有版本可直接比较；并按血缘检查「后版本胜率/场均得分不得低于
  前版本」，否则不通过。

用法：
    python version_benchmark.py --games-per-version 6
    python version_benchmark.py --ckpts "teacher47_v1,teacher47_v2,teacher47_v3"
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cpu_only  # noqa: F401  —— 必须先于 torch
import numpy as np
import torch

from arena_elo import (Agent, HanchanAgent, V1Agent, BCAgent, RuleAgent,
                       play_one, CKPT, _rule_override)
from mahjong.net import MahjongNet
from mahjong import obs2 as O2
from mahjong import obs2_v1 as O2V1
from train_reinforce import slot_of as slot_of54, N_ACTIONS as NA54
from cpu_only import set_inference_threads

ROOT = os.path.dirname(os.path.abspath(__file__))
REF_DEFAULT = "teacher47_v3,bc_v1_last,greedy_tenpai"


def detect_arch(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    if "policy.weight" in sd:
        n = int(sd["policy.weight"].shape[0])
        return {47: "hanchan47", 54: "v1_54", 243: "bc_243"}.get(n, "hanchan47")
    if "trunk.0.weight" in sd:
        return "hand_predictor"
    return "opponent_model"


class AutoAgent(Agent):
    """按 ckpt 里的 channels/blocks/n_actions/touch/extra 自动建网的通用 Agent。

    旧实现把 54 槽一律当 V1Agent（默认 160ch/6blk），遇到 64ch/4blk 的
    a2c_r2/r3/r8 等就 size mismatch——必须从 ckpt 读结构。
    """

    def __init__(self, name, ckpt):
        import train_hanchan as TH
        self.name = name
        self.TH = TH
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck)
        n_actions = int(sd["policy.weight"].shape[0])
        ch = int(ck.get("channels", 160))
        nb = int(ck.get("blocks", 6))
        if "head_fc.0.weight" not in sd:
            # 旧格式/非 MahjongNet（如 243 槽 BC 线）：不要硬塞 AutoAgent，
            # 交给 make_agent 按 detect_arch 分派（BCAgent），否则会 KeyError。
            raise ValueError(
                f"{name}: ckpt 无 head_fc 头（{tuple(sd['policy.weight'].shape) if 'policy.weight' in sd else '?'}），"
                f"请用 version_benchmark.make_agent() 分派到对应 Agent（如 BCAgent）")
        # 以 ck.get("in_planes") 为权威依据分派 obs2 版本。两个家族的平面数互斥：
        #   obs2_v1（旧）：22（order=False）/ 46（order=True，22+4*6）
        #   obs2（当前）：29（order=False）/ 53（order=True）
        # 否则旧 47 槽 ckpt（in_planes=22/46）会被误用当前 obs2 构造 29*20+32=610
        # 维 obs 向量，喂给期望 366 维（64*5+46）的 head_fc -> shape mismatch。
        in_planes = int(ck.get("in_planes", 0) or 0)
        if not in_planes and "embed.weight" in sd:
            # 极旧 ckpt 没写 in_planes：embed.weight 形状 (ch, in_planes, 1) 直接给。
            in_planes = int(sd["embed.weight"].shape[1])
        obs_order = bool(ck.get("obs_order", False))
        if not obs_order:
            # 有些 ckpt 只有 order 版平面数却没写 obs_order：由平面数反推。
            if in_planes == 46 or in_planes == 53:
                obs_order = True
        # 按平面数匹配：优先 obs2_v1，匹配不上（含 in_planes 缺失/是新 29/53）
        # 则回退当前 obs2。
        self.OBS = O2
        if in_planes in (O2V1.N_PLANES, O2V1.ObsEncoder(order=True).n_planes):
            self.OBS = O2V1
        legacy = self.OBS is O2V1
        # 从 head_fc 输入维反推特征模式：26=obs2 标量；46=26+touch8+sit12；
        # 32=当前 obs2 标量（新 ckpt 无 touch/sit）。
        head_in = int(sd["head_fc.0.weight"].shape[1])
        in_scalar = head_in - ch * 5
        extra_total = in_scalar - self.OBS.SCALAR_DIM
        if legacy:
            # 旧 obs2_v1 时代：47 槽 ckpt 均为 hanchan 模式（26+touch8+sit12=46）。
            # 用 head_fc 反推出的 in_scalar 为准（=SCALAR_DIM+touch+extra），ckpt
            # 字段（touch_dim/extra_scalar）只作提示；两者不符时按常见配置
            # touch=8、其余给 sit 收敛，避免个别旧 ckpt 字段缺写/写 None 导致维错。
            hint_t = ck.get("touch_dim")
            hint_e = ck.get("extra_scalar", ck.get("sit_dim"))
            if extra_total > 0:
                if hint_t is not None and int(hint_t) + (int(hint_e) if hint_e is not None else 0) == extra_total:
                    touch, extra = int(hint_t), int(hint_e)
                elif hint_t is not None and 0 < int(hint_t) <= extra_total:
                    touch, extra = int(hint_t), extra_total - int(hint_t)
                else:
                    touch, extra = 8, extra_total - 8
            else:
                touch, extra = 0, 0
            self.mode = "hanchan" if (touch or extra) else "obs2"
        else:
            if int(ck.get("touch_dim", 0)) or extra_total > 0:
                touch = int(ck.get("touch_dim", 8))
                extra = int(ck.get("extra_scalar", ck.get("sit_dim", 12)))
                if touch + extra != extra_total:  # ckpt 没写字段时按常见配置推断
                    touch, extra = 8, extra_total - 8
                self.mode = "hanchan"
            else:
                touch, extra = 0, 0
                self.mode = "obs2"
        self.model = MahjongNet(channels=ch, n_blocks=nb, n_actions=n_actions,
                                touch_dim=touch, extra_scalar=extra,
                                in_planes=in_planes or self.OBS.N_PLANES,
                                aux_dim=int(ck.get("aux_dim", 0)),
                                scalar_dim=self.OBS.SCALAR_DIM)
        self.model.load_state_dict(sd)
        self.model.eval()
        self.enc = self.OBS.ObsEncoder(order=obs_order)
        self.hstate = None
        self.n_actions = n_actions
        self.is47 = n_actions == 47
        if self.is47 and self.mode != "hanchan":
            raise ValueError(f"{name}: 47 槽但特征模式=obs2，无法加载")

    def set_game(self, env, hstate=None):
        self.hstate = hstate

    def act(self, env, p, legal):
        ov = _rule_override(legal)
        if ov is not None:
            return ov
        slot_fn = self.TH.slot_of if self.is47 else slot_of54
        if self.mode == "hanchan":
            vec, sh, sit, gu, tf = self.TH.encode(env, self.enc, self.hstate, p)
            slots = [slot_fn(a) for a in legal if slot_fn(a) >= 0]
            legal2 = [a for a in legal if slot_fn(a) >= 0]
            with torch.no_grad():
                logits = self.model(torch.from_numpy(vec).unsqueeze(0),
                                    torch.from_numpy(tf).unsqueeze(0))[0]
        else:
            planes, scalar = self.enc.encode(env, p)
            vec = np.concatenate([planes.ravel(), scalar]).astype(np.float32)
            slots = [slot_fn(a) for a in legal]
            legal2 = list(legal)
            with torch.no_grad():
                logits = self.model(torch.from_numpy(vec).unsqueeze(0))[0]
        lg = logits.reshape(-1).numpy()
        mask = np.full(self.n_actions, -1e9, dtype=np.float32)
        mask[slots] = 0.0
        pick = int((lg + mask).argmax())
        if self.is47:
            return self.TH.resolve_action_for_slot(legal2, slots, pick, lg)
        return legal2[slots.index(pick)] if pick in slots else legal2[0]


def make_agent(name, path=None):
    from mahjong import policies as POL
    if name.startswith("auxd:"):
        from aux_policy import AuxDangerAgent
        base = name.split(":", 1)[1]
        return AuxDangerAgent(name, os.path.join(CKPT, base if base.endswith(".pt") else base + ".pt"))
    if name.startswith("suphxfull"):
        from call_teacher_suphx import SuphxFullAgent
        return SuphxFullAgent(name)
    if name.startswith("suphx"):
        from external_suphx import SuphxAgent
        return SuphxAgent(name)
    if name.startswith("hybridw:"):
        from hybrid_agents import HybridShantenWinDangerAgent
        return HybridShantenWinDangerAgent(name, path)
    if name.startswith("hybridd:"):
        from hybrid_agents import HybridShantenDangerAgent
        return HybridShantenDangerAgent(name, path)
    if name.startswith("hybrid:"):
        from hybrid_agents import HybridShantenAgent
        return HybridShantenAgent(name, path)
    if path is None or name in POL.POLICIES:
        return RuleAgent(name)
    arch = detect_arch(path)
    if arch == "bc_243":
        return BCAgent(name, path)
    if arch in ("hanchan47", "v1_54"):
        return AutoAgent(name, path)
    raise ValueError(f"{name}: 不是可对战的策略模型（{arch}）")


def ckpt_path_of(name):
    """'hybrid:xxx' / 'xxx' -> checkpoints/xxx.pt 路径。"""
    base = name.split(":", 1)[1] if name.startswith(("hybrid:", "hybridd:", "hybridw:")) else name
    return os.path.join(CKPT, base if base.endswith(".pt") else base + ".pt")


def lineage_of(name):
    n = name.replace(".pt", "")
    if n.startswith("a2c_r"):
        return "A2C"
    if n.startswith("dagger"):
        return "DAgger"
    if n.startswith("hc_r"):
        return "HC"
    if n.startswith("bc_v1"):
        return "BC"
    if n.startswith("from_games"):
        return "from_games"
    if n.startswith("teacher47"):
        return "teacher47"
    if n.startswith(("hybrid:", "hybridd:", "hybridw:")):
        return "hybrid"
    if n.startswith("style_rl"):
        return "style"
    if n == "riichi_net":
        return "riichi_net"
    return "other"


def mtime_order(paths):
    return sorted(paths, key=lambda p: os.path.getmtime(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", default="", help="逗号分隔；默认扫 checkpoints/*.pt")
    ap.add_argument("--games-per-version", type=int, default=6)
    ap.add_argument("--refs", default=REF_DEFAULT,
                    help="参照桌（逗号分隔：ckpt 名或规则策略名），取前 3 个")
    ap.add_argument("--out-md", default=os.path.join(
        ROOT, "rules", "VERSION_BENCH_2026-11-05.md"))
    ap.add_argument("--out-json", default=os.path.join(
        ROOT, "rules", "VERSION_BENCH_2026-11-05.json"))
    args = ap.parse_args()

    set_inference_threads(2)
    ref_names = [x.strip() for x in args.refs.split(",") if x.strip()][:3]
    refs = []
    for r in ref_names:
        path = os.path.join(CKPT, r if r.endswith(".pt") else r + ".pt")
        if os.path.exists(path):
            refs.append(make_agent(r.replace(".pt", ""), path))
        else:
            refs.append(make_agent(r))
    ref_titles = [a.name for a in refs]
    print(f"参照桌：{ref_titles}", flush=True)

    if args.ckpts:
        names = [x.strip() for x in args.ckpts.split(",") if x.strip()]
    else:
        names = [os.path.basename(p).replace(".pt", "")
                 for p in sorted(glob.glob(os.path.join(CKPT, "*.pt")))]
    versions = []
    skipped = []
    for n in names:
        base = n.split(":", 1)[1] if n.startswith(("hybrid:", "hybridd:", "hybridw:")) else n
        path = os.path.join(CKPT, base if base.endswith(".pt") else base + ".pt")
        if not os.path.exists(path):
            skipped.append((n, "不存在"))
            continue
        try:
            arch = detect_arch(path)
            if arch in ("hand_predictor", "opponent_model"):
                skipped.append((n, arch))
                continue
            versions.append((n, path))
        except Exception as e:
            skipped.append((n, f"载入失败 {e}"))
    print(f"待测 {len(versions)} 个可对战版本；跳过 {len(skipped)} 个", flush=True)
    if skipped:
        print("跳过列表：", skipped[:10], flush=True)

    stats = {}
    ref_stats = {a.name: {"games": 0, "wins": 0, "pts": 0.0, "dealin": 0}
                 for a in refs}
    for name, path in versions:
        agent = make_agent(name, path)
        st = {"games": 0, "wins": 0, "pts": 0.0, "dealin": 0}
        for g in range(args.games_per_version):
            seat = g % 4
            table = list(refs)
            # 参照桌固定 3 人，空出的座位给待测版本
            agents4 = []
            ri = 0
            for pos in range(4):
                if pos == seat:
                    agents4.append(agent)
                else:
                    agents4.append(table[ri])
                    ri += 1
            out = play_one(agents4, seed=20261105 + g * 131)
            if out is None:
                continue
            delta, winner, _aux, _ad, dealin = out
            st["games"] += 1
            st["pts"] += delta[seat]
            if winner == seat:
                st["wins"] += 1
            if dealin[seat]:
                st["dealin"] += 1
            # 同一批对局里累计参照桌读数（直接对照「新的厉害的」）
            for pos, ag in enumerate(agents4):
                if pos == seat:
                    continue
                r = ref_stats[ag.name]
                r["games"] += 1
                r["pts"] += delta[pos]
                if winner == pos:
                    r["wins"] += 1
                if dealin[pos]:
                    r["dealin"] += 1
        g = max(1, st["games"])
        st.update({"win_rate": round(100.0 * st["wins"] / g, 1),
                   "avg_pt": round(st["pts"] / g, 1),
                   "deal_in_rate": round(100.0 * st["dealin"] / g, 1),
                   "arch": detect_arch(path)})
        stats[name] = st
        print(f"  {name:<24} 和牌率 {st['win_rate']:5.1f}%  "
              f"场均 {st['avg_pt']:+7.1f}  放铳 {st['deal_in_rate']:4.1f}%",
              flush=True)

    ref_out = {}
    for k, v in ref_stats.items():
        g = max(1, v["games"])
        ref_out[k] = {"games": v["games"],
                      "win_rate": round(100.0 * v["wins"] / g, 1),
                      "avg_pt": round(v["pts"] / g, 1),
                      "deal_in_rate": round(100.0 * v["dealin"] / g, 1)}

    # ---- 血缘内后版本 vs 前版本（胜率与场均都不得低于前版本）----
    by_lin = {}
    for name, st in stats.items():
        by_lin.setdefault(lineage_of(name), []).append((name, st))
    gates = []
    for lin, lst in by_lin.items():
        lst.sort(key=lambda x: os.path.getmtime(ckpt_path_of(x[0])))
        for k in range(1, len(lst)):
            prev, cur = lst[k - 1], lst[k]
            # 用户要求「后版本必须比前版本强」：等价不算过（HOLD），
            # 任意一项更差算 FAIL，两项都不低且至少一项严格更高才 PASS。
            pw, cw = prev[1]["win_rate"], cur[1]["win_rate"]
            pp, cp = prev[1]["avg_pt"], cur[1]["avg_pt"]
            if cw < pw or cp < pp:
                verdict = "FAIL"
            elif cw == pw and cp == pp:
                verdict = "HOLD"      # 一样强不算「更强」，不给过
            else:
                verdict = "PASS"
            gates.append({"lineage": lin, "prev": prev[0], "cur": cur[0],
                          "prev_win": prev[1]["win_rate"],
                          "cur_win": cur[1]["win_rate"],
                          "prev_pt": prev[1]["avg_pt"],
                          "cur_pt": cur[1]["avg_pt"],
                          "verdict": verdict})

    # ---- Markdown ----
    lines = ["# 版本基准赛：所有版本同强桌对比（2026-11-05）", "",
             f"> 参照桌固定：**{ ' / '.join(ref_titles) }**（新强版本 + 人类牌谱"
             f"教师 + 强规则基线）；每个版本轮流坐 4 个座位，各打 "
             f"{args.games_per_version} 局。", "",
             "> 读数：和牌率 = 该版本坐庄局里和牌占比；场均得分 = 每局净点数；"
             "放铳率 = 该版本放铳局占比。同参照桌下可比；「后版本必须比前版本强"
             "（胜率与场均都不低）」的判定见文末。", "",
             "| 参照 | 局数 | 和牌率 | 场均得分 | 放铳率 |",
             "|---|---|---|---|---|"]
    for k, v in ref_out.items():
        lines.append(f"| **{k}**（参照） | {v['games']} | {v['win_rate']}% | "
                     f"{v['avg_pt']:+.1f} | {v['deal_in_rate']}% |")
    lines += ["",
             "## 全部版本读数", "",
             "| 版本 | 血缘 | 架构 | 局数 | 和牌率 | 场均得分 | 放铳率 |",
             "|---|---|---|---|---|---|---|"]
    for name, st in stats.items():
        lines.append(f"| `{name}.pt` | {lineage_of(name)} | {st['arch']} | "
                     f"{st['games']} | {st['win_rate']}% | {st['avg_pt']:+.1f} | "
                     f"{st['deal_in_rate']}% |")
    lines += ["", "## 晋级判定（同血缘相邻版本）", "",
              "| 血缘 | 前版本 → 后版本 | 胜率(前→后) | 场均(前→后) | 判定 |",
              "|---|---|---|---|---|"]
    for g in gates:
        lines.append(f"| {g['lineage']} | `{g['prev']}` → `{g['cur']}` | "
                     f"{g['prev_win']}%→{g['cur_win']}% | "
                     f"{g['prev_pt']:+.1f}→{g['cur_pt']:+.1f} | "
                     f"**{g['verdict']}** |")
    passed = sum(1 for g in gates if g["verdict"] == "PASS")
    lines += ["", f"结论：{passed}/{len(gates)} 个相邻版本通过"
              "（胜率与场均都不低于前版本）。", "",
              "注：每版局数少（默认 6 局），胜率分辨率约 ±16pt，只作**版本间"
              "粗筛**；要下定论用 version_ladder.py 的 95% CI 闸门。"]
    text = "\n".join(lines)
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    payload = {"refs": ref_titles, "ref_stats": ref_out,
               "games_per_version": args.games_per_version,
               "stats": stats, "gates": gates}
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print("\n" + text, flush=True)
    print(f"写入 {args.out_md}")
    print(f"写入 {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())