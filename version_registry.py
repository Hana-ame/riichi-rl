"""版本登记表：把 checkpoints/ 下所有模型版本登记成一张表（Markdown + JSON）。

用户要求（2026-11-05）：「都记录一下各个版本」。
本脚本自动扫描 checkpoint，提取架构/参数量/训练元数据/赛评读数/天梯闸门判定，
生成 `rules/VERSION_REGISTRY_2026-11-05.md` 与同名 `.json`。

用法：
    python version_registry.py                      # 默认写 rules/VERSION_REGISTRY_2026-11-05.*
    python version_registry.py --out rules/X.md --json rules/X.json
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
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(ROOT, "checkpoints")

#: ELO.json 里的人名 -> ckpt 文件名（用于挂最新赛评读数）
ELO_NAME = {
    "riichi_net.pt": "riichi_net(纯RL)",
    "from_games_v1.pt": "from_games_v1(蒸馏)",
    "from_games_v2.pt": "from_games_v2(蒸馏)",
    "a2c_r10_it25.pt": "a2c_r10(半庄A2C)",
    "dagger_it120.pt": "dagger(半庄DAgger)",
    "hc_r13_it55.pt": "hc(半庄HC)",
    "bc_v1_last.pt": "bc_v1(人类牌谱教师)",
    "teacher47_v3.pt": "teacher47_v3(加权BC)",
}

#: 已知的自对弈评估读数（eval_hanchan；来源见对应 rules/*.md）
EVAL_NOTE = {
    "teacher47_v1": "自对弈 5.0%（40 局）",
    "teacher47_v2": "自对弈 15.0%（80 局）",
    "teacher47_v3": "自对弈 17.5%（80 局）",
    "style_rl_v1": "自对弈 7.5%（40 局，退化）",
}


def lineage_of(name):
    """文件名 -> 血缘名。"""
    n = name.replace(".pt", "")
    if n.startswith("a2c_r"):
        return "A2C（早期 RL）"
    if n.startswith("dagger"):
        return "DAgger"
    if n.startswith("hc_r"):
        return "Hanchan HC"
    if n.startswith("bc_v1"):
        return "BC 人类牌谱教师"
    if n.startswith("from_games"):
        return "蒸馏（from_games）"
    if n.startswith("teacher47"):
        return "教师 BC 固定槽（teacher47）"
    if n.startswith("style_rl"):
        return "风格微调（style）"
    if n == "riichi_net":
        return "纯 RL（riichi_net）"
    if n.startswith("opponent"):
        return "对手手牌推断"
    if n.startswith("hand_predictor"):
        return "和牌预测器"
    return "其他"


def arch_of(sd):
    """按 state_dict 结构判断架构。"""
    if "policy.weight" in sd:
        n = int(sd["policy.weight"].shape[0])
        return {47: "hanchan47", 54: "convformer54", 243: "bc243"}.get(
            n, f"policy{n}")
    if "head_fc.0.weight" in sd and "policy.weight" in sd:
        return "policy"
    # 预测模型（不输出动作）
    if "trunk.0.weight" in sd and "win.weight" in sd:
        return "hand_predictor"
    if "net.0.weight" in sd and "net.4.weight" in sd:
        return "opponent_model"
    if "model.state_dict" in sd:
        return "packed"
    keys = list(sd.keys())[:3]
    return "unknown(" + ",".join(keys) + ")"


def count_params(sd):
    return sum(int(v.numel()) for v in sd.values()
               if torch.is_tensor(v))


def load_meta(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    meta = {k: v for k, v in ck.items()
            if k != "state_dict" and isinstance(v, (int, float, str, bool))}
    return ck, sd, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=CKPT)
    ap.add_argument("--out", default=os.path.join(
        ROOT, "rules", "VERSION_REGISTRY_2026-11-05.md"))
    ap.add_argument("--json", default=os.path.join(
        ROOT, "rules", "VERSION_REGISTRY_2026-11-05.json"))
    args = ap.parse_args()

    elo = {}
    elo_path = os.path.join(ROOT, "rules", "ELO.json")
    if os.path.exists(elo_path):
        for r in json.load(open(elo_path, encoding="utf-8")):
            elo[r["name"]] = r

    bench = {}
    bench_refs = []
    bp = os.path.join(ROOT, "rules", "VERSION_BENCH_2026-11-05.json")
    if os.path.exists(bp):
        bd = json.load(open(bp, encoding="utf-8"))
        bench = bd.get("stats", {})
        bench_refs = bd.get("refs", [])
        bench_ref_stats = bd.get("ref_stats", {})
    else:
        bench_ref_stats = {}
    bench_gate = {}
    if os.path.exists(bp):
        for g in bd.get("gates", []):
            bench_gate[g["cur"]] = g

    gates = {}
    for gname in ("VERSION_LADDER_teacher47.json", "VERSION_LADDER_a2c.json",
                  "VERSION_LADDER_from_games.json"):
        gp = os.path.join(ROOT, "rules", gname)
        if not os.path.exists(gp):
            continue
        d = json.load(open(gp, encoding="utf-8"))
        for g in d.get("gates", []):
            gates[g["cur"]] = g

    rows = []
    for path in sorted(glob.glob(os.path.join(args.dir, "*.pt"))):
        name = os.path.basename(path)
        base = name.replace(".pt", "")
        try:
            ck, sd, meta = load_meta(path)
            arch = arch_of(sd)
            n_params = count_params(sd)
        except Exception as e:  # 坏文件也要登记，不能悄悄跳过
            arch, n_params, meta = "BROKEN", 0, {"error": str(e)}
        stat = os.stat(path)
        row = {
            "file": name,
            "lineage": lineage_of(name),
            "arch": arch,
            "params": n_params,
            "size_mb": round(stat.st_size / 1024 / 1024, 2),
            "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)),
            "meta": meta,
            "eval": EVAL_NOTE.get(base, ""),
            "elo": None,
            "gate": None,
            "bench": bench.get(base),
            "bench_gate": bench_gate.get(base),
            "status": "历史版本（无闸门记录）",
        }
        elo_name = ELO_NAME.get(name)
        if elo_name and elo_name in elo:
            r = elo[elo_name]
            row["elo"] = {"name": r["name"], "elo": r["elo"],
                          "win_rate": r["win_rate"], "deal_in_rate":
                          r.get("deal_in_rate"), "avg_pt": r["avg_pt"]}
            if (r.get("win_rate") or 0) >= 15.0:
                row["status"] = "在用（和牌率达标）"
        g = gates.get(base)
        if g:
            row["gate"] = g
            row["status"] = {"PASS": "通过闸门（可发布）",
                             "FAIL": "未通过闸门（不发布）",
                             "HOLD": "证据不足（HOLD）"}[g["verdict"]]
        rows.append(row)

    # 每个血缘内按 mtime 编版本号 V1..Vn
    # A2C 血缘早期混用 54/47 两套动作空间，版本号不能混编——按架构拆开
    for r in rows:
        if r["lineage"].startswith("A2C"):
            r["lineage"] = ("A2C·47槽（早期 RL）" if r["arch"] == "hanchan47"
                            else "A2C·54槽（早期 RL）")
    by_lineage = {}
    for r in rows:
        by_lineage.setdefault(r["lineage"], []).append(r)
    for lin, lst in by_lineage.items():
        lst.sort(key=lambda r: r["mtime"])
        for k, r in enumerate(lst, 1):
            r["version"] = f"V{k}"

    order = ["纯 RL（riichi_net）", "A2C·47槽（早期 RL）",
             "A2C·54槽（早期 RL）", "DAgger", "Hanchan HC",
             "BC 人类牌谱教师", "蒸馏（from_games）",
             "教师 BC 固定槽（teacher47）", "风格微调（style）",
             "对手手牌推断", "和牌预测器", "其他"]
    by_lineage = dict(sorted(by_lineage.items(),
                             key=lambda kv: order.index(kv[0])
                             if kv[0] in order else 99))

    # ---- Markdown ----
    lines = ["# 模型版本登记表（2026-11-05）", "",
             f"> 自动扫描 `checkpoints/`：共 **{len(rows)}** 个模型文件、"
             f"**{len(by_lineage)}** 条血缘。", "",
             "> 字段：版本（同血缘内按 mtime 排序）/ ckpt / 架构与参数量 /"
             " 训练元数据 / 赛评读数 / 状态（天梯闸门优先）。"]
    if bench:
        refs_txt = " / ".join(
            f"{k} 和牌率 {v['win_rate']}% 场均 {v['avg_pt']:+.0f}"
            for k, v in bench_ref_stats.items())
        lines += ["", f"> 强桌基准：版本与**{ ' / '.join(bench_refs) }**同席对局"
                  f"（{bench_ref_stats and '参照读数：' + refs_txt}），"
                  "表内「强桌」列为该版本在这些对局里的和牌率/场均得分。"]
    lines.append("")
    for lin, lst in by_lineage.items():
        lines += [f"## {lin}（{len(lst)} 个）", "",
                  "| 版本 | ckpt | 架构 | 参数量 | 训练元数据 | 赛评读数 |"
                  " 强桌 和牌率/场均 | 状态 |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in lst:
            m = r["meta"]
            if r["arch"] == "hand_predictor":
                meta_s = f"samples={m.get('samples')} base_win={m.get('base_win')}"
            elif r["arch"] == "opponent_model":
                meta_s = f"samples={m.get('samples')} val_loss={m.get('val_loss')}"
            else:
                parts = []
                for k in ("iter", "deals", "win_rate", "epoch", "val_acc",
                          "channels", "blocks"):
                    if k in m:
                        parts.append(f"{k}={m[k]}")
                meta_s = " ".join(parts) or "-"
            ev = r["eval"] or "-"
            if r["elo"]:
                e = r["elo"]
                ev += (f"；ELO {e['elo']}（win {e['win_rate']}%, "
                       f"放铳 {e.get('deal_in_rate')}%, pts {e['avg_pt']:+}）")
            status = r["status"]
            if r["gate"]:
                g = r["gate"]
                status += (f"（点差 {g['mean_diff']:+.0f} "
                           f"[{g['ci_lo']:+.0f},{g['ci_hi']:+.0f}]）")
            b = r.get("bench")
            btxt = (f"{b['win_rate']}% / {b['avg_pt']:+.0f}（{b['games']}局）"
                    if b else "未测")
            if r.get("bench_gate"):
                bg = r["bench_gate"]
                btxt += f" → 基准闸门 {bg['verdict']}"
            lines.append(f"| {r['version']} | `{r['file']}` | {r['arch']} | "
                         f"{r['params']:,} | {meta_s} | {ev} | {btxt} | "
                         f"{status} |")
        lines.append("")
    lines += ["## 说明", "",
              "- 版本号只在**同一血缘内部**排序，跨血缘不可比。",
              "- 状态优先级：天梯闸门（PASS/FAIL/HOLD） > ELO 和牌率是否达标"
              "（≥15%）> 历史版本。",
              "- 天梯闸门来自 `version_ladder.py` 的规则：相邻版本同局净点差"
              " 95% CI 下界 >0 且 Elo 未显著下滑才 PASS。",
              f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M')}；"
              "权重文件本身不入库（`.gitignore` 排除 `*.pt`）。"]
    text = "\n".join(lines)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump({"generated": time.time(), "rows": rows}, f,
                  ensure_ascii=False, indent=1)
    print(text)
    print(f"\n写入 {args.out}")
    print(f"写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())