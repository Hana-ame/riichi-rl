"""Elo 竞技场：所有模型/规则策略同台对局，按点数排名更新 Elo，筛选最优策略。

参赛者（自动识别架构加载）：
  规则基线   random / greedy_tenpai / call_first / yaochu_first
  54 槽线    riichi_net（纯 RL）、from_games_v1/v2（蒸馏）
  47 槽线    a2c_r10_it25 / dagger_it120 / hc_r13_it55（半庄 A2C/DAgger/HC）
  243 槽线   bc_v1_last（BC 教师，人类牌谱——单独标注）

统一规则：所有神经网络 Agent 都做「和牌/立直规则接管」（纯网络 argmax 永远不选
占比很低的立直槽，实测 0 和牌）；规则基线自带同等逻辑。

用法：
    python arena_elo.py --games 300 --out rules/ELO.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np             # noqa: E402
import cpu_only                # noqa: E402,F401 —— 必须先于 torch 锁 CPU-only
import torch                   # noqa: E402

from mahjong import aux_tasks as AUX
from mahjong import env as E
from mahjong import obs2 as O2
from mahjong import policies as POL
from mahjong.hanchan import HanchanState
from mahjong.net import MahjongNet
from train_reinforce import slot_of as slot_of54, N_ACTIONS as NA54

ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(ROOT, "checkpoints")


def _rule_override(legal):
    """和牌 > 立直 的统一规则接管。"""
    for a in legal:
        if a.type in (E.ACT_TSUMO, E.ACT_RON):
            return a
    for a in legal:
        if a.type == E.ACT_RIICHI:
            return a
    return None


class Agent:
    name = "base"
    kind = "base"

    def act(self, env, p, legal):
        raise NotImplementedError

    def set_game(self, env, hstate=None):
        pass


class RuleAgent(Agent):
    kind = "rule"

    def __init__(self, name, label=None):
        self.name = label or name
        self.pol = POL.make_policy(name, seed=0)

    def act(self, env, p, legal):
        return self.pol.choose(env, p, legal)


class V1Agent(Agent):
    """54 槽 convformer_v1（mahjong.net.MahjongNet 默认 160/6，obs2 774 维）。"""
    kind = "v1_54"

    def __init__(self, name, ckpt):
        self.name = name
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model = MahjongNet()
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.enc = O2.ObsEncoder()

    def act(self, env, p, legal):
        ov = _rule_override(legal)
        if ov is not None:
            return ov
        planes, scalar = self.enc.encode(env, p)
        vec = np.concatenate([planes.ravel(), scalar]).astype(np.float32)
        slots = [slot_of54(a) for a in legal]
        with torch.no_grad():
            logits = self.model(torch.from_numpy(vec).unsqueeze(0))[0].reshape(-1).numpy()
        mask = np.full(NA54, -1e9, dtype=np.float32)
        mask[slots] = 0.0
        pick = int((logits + mask).argmax())
        return legal[slots.index(pick)] if pick in slots else legal[0]


class HanchanAgent(Agent):
    """47 槽 hanchan 线（64ch/4blk，obs2+touch8+sit12 = 794 维）。"""
    kind = "hanchan_47"

    def __init__(self, name, ckpt):
        import train_hanchan as TH
        self.name = name
        self.TH = TH
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model = MahjongNet(channels=ck.get("channels", 64),
                                n_blocks=ck.get("blocks", 4),
                                n_actions=47, touch_dim=8, extra_scalar=12)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.enc = O2.ObsEncoder()
        self.hstate = None

    def set_game(self, env, hstate=None):
        self.hstate = hstate

    def act(self, env, p, legal):
        ov = _rule_override(legal)
        if ov is not None:
            return ov
        vec, sh, sit, give_up, tf = self.TH.encode(env, self.enc, self.hstate, p)
        slots = [self.TH.slot_of(a) for a in legal if self.TH.slot_of(a) >= 0]
        legal2 = [a for a in legal if self.TH.slot_of(a) >= 0]
        if not slots:
            return legal[0]
        with torch.no_grad():
            logits, values = self.model(
                torch.from_numpy(vec).unsqueeze(0),
                torch.from_numpy(tf).unsqueeze(0))
        lg = logits[0].numpy()
        mask = np.full(47, -1e9, dtype=np.float32)
        mask[slots] = 0.0
        pick = int((lg + mask).argmax())
        return legal2[slots.index(pick)] if pick in slots else legal2[0]


class BCAgent(Agent):
    """243 槽 BC 线（96ch/4blk，16 平面×34+12 标量，人类牌谱教师——单独标注）。"""
    kind = "bc_243"

    def __init__(self, name, ckpt):
        import arena_model as AM
        from train_bc import MahjongNet as BCNet
        self.name = name
        self.AM = AM
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model = BCNet(ck.get("channels", 96), ck.get("blocks", 4))
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()

    def act(self, env, p, legal):
        import build_cache as BC
        import houou_data as H
        from action_bridge import env_action_to_slot
        ov = _rule_override(legal)
        if ov is not None:
            return ov
        discards_by = np.zeros((4, 34), dtype=np.float32)
        riichi_flags = np.zeros(4, dtype=np.float32)
        meld_cnt = np.zeros(4, dtype=np.float32)
        turn = max(0, (136 - (len(env.wall) - env.wall_idx)) // 4)
        for i in range(4):
            for t in env.players[i].river:
                discards_by[i, t] += 1
            riichi_flags[i] = 1.0 if env.players[i].riichi else 0.0
            meld_cnt[i] = len(env.players[i].melds)
        pl, sc = self.AM.encode_live(env, p, discards_by, riichi_flags,
                                     meld_cnt, turn, 0, trigger_actor=p,
                                     trigger_tile=-1)
        slots = [env_action_to_slot(a, BC.TILED_SLOT_BASE, BC.PARAM_SLOT,
                                    H.TILE_FIELDS) for a in legal]
        good = [i for i, s in enumerate(slots) if s >= 0]
        if not good:
            return legal[0]
        with torch.no_grad():
            logit, _ = self.model(torch.from_numpy(pl).unsqueeze(0),
                                  torch.from_numpy(sc).unsqueeze(0))
        lg = logit[0].float().numpy()
        bi, bv = good[0], -1e18
        for i in good:
            if lg[slots[i]] > bv:
                bv, bi = lg[slots[i]], i
        return legal[bi]


# ---------- 参赛者注册 ----------

def build_participants():
    P = []
    # 立直策略族：全部在同一批对局里比点数（用户要求「不同立直策略/默听策略，
    # 要求分最大」——由 Elo 和平均点数直接判谁点最多）。
    rule_agents = [
        ("random", None),
        ("greedy_tenpai", "greedy_tenpai(立直优先)"),
        ("damaten", "damaten(默听)"),
        ("riichi_needed", "riichi_needed(没役才立直)"),
        ("riichi_ev", "riichi_ev(按点数期望)"),
        ("call_first", None),
        ("yaochu_first", None),
    ]
    for n, label in rule_agents:
        P.append(RuleAgent(n, label))

    def add(cls, name, fname, **kw):
        path = os.path.join(CKPT, fname)
        if not os.path.exists(path):
            print(f"[skip] {fname} 不存在", flush=True)
            return
        try:
            P.append(cls(name, path, **kw))
            print(f"[load] {name} <- {fname}", flush=True)
        except Exception as e:
            print(f"[skip] {name} 加载失败: {e}", flush=True)

    # V1Agent 是 54 槽旧架构，与本 arena 47 槽模型不兼容（logits 54 vs 47），
    # 2026-11-24 实测触发 ValueError。需要时用独立 54 槽 arena。
    # add(V1Agent, "riichi_net(纯RL)", "riichi_net.pt")
    # add(V1Agent, "from_games_v1(蒸馏)", "from_games_v1.pt")
    # add(V1Agent, "from_games_v2(蒸馏)", "from_games_v2.pt")
    add(HanchanAgent, "a2c_r10(半庄A2C)", "a2c_r10_it25.pt")
    add(HanchanAgent, "dagger(半庄DAgger)", "dagger_it120.pt")
    add(HanchanAgent, "hc(半庄HC)", "hc_r13_it55.pt")
    add(HanchanAgent, "teacher47_v3(加权BC)", "teacher47_v3.pt")
    add(HanchanAgent, "teacher47_100h_v2(最优BC)", "teacher47_100h_v2.pt")
    # bc_v1 是 54 槽旧架构，与本 arena 其他 47 槽模型不兼容（logits 54 vs 47），
    # 2026-11-24 实测触发 ValueError，已剔除。需要时用独立 54 槽 arena。
    # add(BCAgent, "bc_v1(人类牌谱教师)", "bc_v1_last.pt")
    # 新增：eff_uke 系（2026-11-06 牌效教师线）
    add(HanchanAgent, "eff_wuke_d2(牌效d2)", "teacher47_eff_wuke_d2.pt")
    add(HanchanAgent, "tenhou_bc(Tenhou人类)", "tenhou_bc.pt")
    return P


# ---------- 对局与 Elo ----------

def play_one(agents, seed, env_cls=HanchanState, east_only=False):
    h = env_cls(dealer=seed % 4, east_only=east_only)
    env = h.new_env(seed=seed)
    seats = list(range(4))
    for i, a in enumerate(agents):
        a.set_game(env, h)
    # 每个模型都领一份「额外任务清单」（用户要求：完成也有奖励）
    trackers = [AUX.AuxTaskTracker(i) for i in range(4)]
    r = env.start_turn()
    steps = 0
    while r is None and steps < 600:
        steps += 1
        for tr in trackers:
            tr.observe(env)
        acts = {}
        for p in env.pending_players():
            legal = env.legal_actions(p)
            if not legal:
                acts[p] = E.PASS_ACTION
                continue
            acts[p] = agents[p].act(env, p, legal)
        r = env.step(acts)
    if r is None:
        return None
    # 排名的点数必须用 result.scores（绝对点数）：points_delta 只是本局
    # 转账流水，不含立直时直接扣走的 1000 点供托。
    delta = E.net_points(r, env.initial_points)
    winner = -1
    if r.result_type != "ryuukyoku" and r.winner is not None and r.winner >= 0:
        winner = r.winner
    aux_bonus = []
    aux_done = []
    for tr in trackers:
        done, bonus = tr.finish(env, r)
        aux_done.append(done)
        aux_bonus.append(bonus)
    deal_in = [r.result_type == "ron" and r.from_player == pos
               for pos in range(4)]
    return delta, winner, aux_bonus, aux_done, deal_in


def elo_update(ratings, sel, deltas, K=24.0):
    """sel[pos] = 该座位参赛者在池中的索引；deltas[pos] = 座位 pos 的点数。

    零和修正：**逐对**用「名次比较」的 1/0.5/0 计分，而不是拿
    [0.75,0.5,0.25,0] 的名次分当逐对得分。旧实现里一对 (第1,第4) 的
    s_a+s_b = 0.75 ≠ 1，导致每局总分不守恒、Elo 整体漂移
    （实测 200 局后全场均值从 1500 掉到 ~870，排名本身看似可用但刻度错了）。
    """
    order = sorted(range(4), key=lambda pos: -deltas[pos])
    rank_of = {pos: rank for rank, pos in enumerate(order)}
    for i in range(4):
        for j in range(i + 1, 4):
            a, b = sel[i], sel[j]
            ra, rb = rank_of[i], rank_of[j]
            sa = 1.0 if ra < rb else (0.5 if ra == rb else 0.0)
            ea = 1.0 / (1.0 + 10 ** ((ratings[b] - ratings[a]) / 400.0))
            ratings[a] += K * (sa - ea)
            ratings[b] += K * ((1.0 - sa) - (1.0 - ea))


def render_markdown(rows, n_games, aux_weight=1.0, selection_note=""):
    """把排名字典列表渲染成 ELO.md（纯函数，便于从 ELO.json 重新生成）。

    rows 若带 avg_bonus（额外任务奖励），表里多一列并附任务完成率汇总。
    """
    games = [r["games"] for r in rows if r.get("games")]
    gmin, gmax = (min(games), max(games)) if games else (0, 0)
    mean_g = (sum(games) / len(games)) if games else 1
    # 在线 Elo 的标准误粗估：400/sqrt(n)；保守取 2×SE 作为「不可区分」带宽
    se = 400.0 / (mean_g ** 0.5) if mean_g else 0.0
    band = 2 * se
    has_aux = any("avg_bonus" in r for r in rows)
    lines = ["# Elo 竞技场：所有模型/策略排名", "",
             f"> {n_games} 局，4 人桌随机组合，按最终点数排名更新 Elo"
             "（K=24，初始 1500，逐对零和更新）。",
             "> 统一规则接管：所有神经网络 Agent 的和牌/立直由规则保证。"]
    if has_aux:
        lines.append(f"> 额外任务：每个参赛者每局领一份任务清单（和牌/自摸/"
                     f"立直/满贯/未放铳/流局听牌/8 巡内进听），完成即加奖励；"
                     f"奖励按权重 {aux_weight:g} 计入 Elo 排名。")
    lines.append("> 选员：先全体热身，之后每轮保留 Elo 前一半 + 随机几个替换"
                 "（表现差的会被换下；随机替换保证长尾仍有对局）。")
    if selection_note:
        lines.append(f"> 最后一次选员：{selection_note}")
    lines += [f"> 判读：每人 {gmin}–{gmax} 局，Elo 标准误粗估 ±{se:.0f}；"
              f"**相邻名次差 <{band:.0f} 不能当结论**（K=24 收敛慢，"
              "至少 `--games 600` 再说）。", ""]
    has_dealin = any("deal_in_rate" in r for r in rows)
    head = ("| 排名 | 参赛者 | 类型 | Elo | 局数 | 和牌率 | "
            + ("放铳率 | " if has_dealin else "")
            + "平均点数 |" + (" 任务奖励/局 |" if has_aux else ""))
    sep = "|---" * (8 if has_dealin and has_aux else
                    (7 if has_dealin or has_aux else 6)) + "|"
    lines += [head, sep]
    for k, r in enumerate(rows, 1):
        line = (f"| {k} | {r['name']} | {r['kind']} | **{r['elo']}** | "
                f"{r['games']} | {r['win_rate']}% |")
        if has_dealin:
            line += f" {r.get('deal_in_rate', 0):.1f}% |"
        line += f" {r['avg_pt']:+.0f} |"
        if has_aux:
            line += f" {r.get('avg_bonus', 0):+.0f} |"
        lines.append(line)

    if has_aux and rows and rows[0].get("aux_tasks"):
        total_deals = sum(r["games"] for r in rows)
        names = list(rows[0]["aux_tasks"].keys())
        lines += ["", "## 额外任务完成率（每人每局）", "",
                  "| 参赛者 | " + " | ".join(names) + " |",
                  "|---" * (len(names) + 1) + "|"]
        for r in rows:
            g = max(1, r["games"])
            cells = [f"{100.0 * r['aux_tasks'].get(n, 0) / g:.1f}%"
                     for n in names]
            lines.append(f"| {r['name']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "rules", "ELO.md"))
    ap.add_argument("--aux-weight", type=float, default=1.0,
                    help="额外任务奖励计入排名/Elo 的权重（0=不计）")
    ap.add_argument("--warmup-games", type=int, default=100,
                    help="先全体参赛热身 N 局，之后按「前一半+随机替换」选员")
    ap.add_argument("--reselect-every", type=int, default=100,
                    help="每 N 局重新选员（保留 Elo 前一半 + 随机几个替换）")
    ap.add_argument("--random-fill", type=int, default=3,
                    help="选员时从落选者里随机替换几个进参赛名单")
    ap.add_argument("--mode", default="hanchan", choices=["hanchan", "east"],
                    help="hanchan=半庄 E1-S4，east=东风战 E1-E4")
    args = ap.parse_args()

    from cpu_only import set_inference_threads
    set_inference_threads(2)
    rng = random.Random(args.seed)
    agents = build_participants()
    names = [a.name for a in agents]
    ratings = {i: 1500.0 for i in range(len(agents))}
    games_played = {i: 0 for i in range(len(agents))}
    wins = {i: 0 for i in range(len(agents))}
    pt_sum = {i: 0.0 for i in range(len(agents))}
    aux_sum = {i: 0.0 for i in range(len(agents))}
    aux_count = {i: {name: 0 for name in AUX.TASKS} for i in range(len(agents))}
    deal_in_cnt = {i: 0 for i in range(len(agents))}
    active = list(range(len(agents)))
    selection_note = ""
    t0 = time.time()

    def reselect():
        """选员：Elo 前一半 + 从落选者里随机替换几个。"""
        order = sorted(range(len(agents)), key=lambda i: -ratings[i])
        keep_n = max(4, len(agents) // 2)
        keep = order[:keep_n]
        rest = order[keep_n:]
        fill = rng.sample(rest, min(args.random_fill, len(rest))) if rest else []
        chosen = keep + fill
        names = ", ".join(agents[i].name for i in sorted(chosen))
        return chosen, f"{time.time() - t0:.0f}s 选员：保留 Elo 前 {keep_n} 名 + 随机 {len(fill)} 个替换 -> {names}"

    for g in range(args.games):
        if g == args.warmup_games and args.games > args.warmup_games:
            active, selection_note = reselect()
            print("[arena] " + selection_note, flush=True)
        elif g > args.warmup_games and (g - args.warmup_games) % max(1, args.reselect_every) == 0:
            active, selection_note = reselect()
            print("[arena] " + selection_note, flush=True)
        pool = active if len(active) >= 4 else list(range(len(agents)))
        sel = rng.sample(pool, 4)
        picked = [agents[i] for i in sel]
        out = play_one(picked, seed=args.seed * 7919 + g * 131,
                       east_only=(args.mode == "east"))
        if out is None:
            continue
        delta, winner, aux_bonus, aux_done, dealin = out
        # 额外任务奖励计入排名/Elo（用户要求：完成也有奖励）
        ranked = [delta[pos] + args.aux_weight * aux_bonus[pos]
                  for pos in range(4)]
        elo_update(ratings, sel, ranked)
        for pos, i in enumerate(sel):
            games_played[i] += 1
            pt_sum[i] += delta[pos]
            aux_sum[i] += aux_bonus[pos]
            for name, ok in aux_done[pos].items():
                if ok:
                    aux_count[i][name] += 1
            if winner == pos:
                wins[i] += 1
            if dealin[pos]:
                deal_in_cnt[i] += 1
        if (g + 1) % 25 == 0:
            print(f"[arena] {g+1}/{args.games} games "
                  f"({time.time()-t0:.0f}s)", flush=True)

    rows = []
    for i, name in enumerate(names):
        rows.append({
            "name": name,
            "kind": agents[i].kind,
            "elo": round(ratings[i], 1),
            "games": games_played[i],
            "win_rate": round(wins[i] / max(1, games_played[i]) * 100, 1),
            "avg_pt": round(pt_sum[i] / max(1, games_played[i]), 1),
            "avg_bonus": round(aux_sum[i] / max(1, games_played[i]), 1),
            "deal_in_rate": round(deal_in_cnt[i]
                                  / max(1, games_played[i]) * 100, 1),
            "aux_tasks": dict(aux_count[i]),
        })
    rows.sort(key=lambda r: -r["elo"])

    text = render_markdown(rows, args.games, args.aux_weight,
                           selection_note)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(text + "\n")
    print("\n" + text, flush=True)
    with open(os.path.join(ROOT, "rules", "ELO.json"), "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())