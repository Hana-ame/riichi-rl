"""危险张验证：hybrid vs hybridd 同局对照，统计放铳率是否真的下降。

背景：hybridd（代码向听 + 危险张）在 20 局里只看到场均提升、放铳率没变。
本脚本把 hybrid 与 hybridd 放进**同一批牌局**（每局两张桌位都占，座位轮换），
用 ≥200 局做大样本：比较放铳率（两比例差 + 95% CI）、和牌率、场均得分。

用法：
    python danger_validate.py --games 240
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cpu_only  # noqa: F401
import torch

from arena_elo import RuleAgent, play_one, CKPT
from cpu_only import set_inference_threads
from version_benchmark import make_agent

ROOT = os.path.dirname(os.path.abspath(__file__))
SPEC_A = "hybrid:teacher47_v3"
SPEC_B = "hybridd:teacher47_v3"


def two_prop_ci(k1, n1, k2, n2):
    """两比例差 (k2/n2 - k1/n1) 与 95% CI（正态近似）。"""
    if n1 == 0 or n2 == 0:
        return 0.0, 0.0, 0.0
    p1, p2 = k1 / n1, k2 / n2
    d = p2 - p1
    se = math.sqrt(max(1e-12, p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2))
    return d, d - 1.96 * se, d + 1.96 * se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=240)
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--out-md", default=os.path.join(
        ROOT, "rules", "DANGER_VALIDATION_2026-11-05.md"))
    ap.add_argument("--out-json", default=os.path.join(
        ROOT, "rules", "DANGER_VALIDATION_2026-11-05.json"))
    args = ap.parse_args()

    set_inference_threads(2)
    a = make_agent(SPEC_A, os.path.join(CKPT, "teacher47_v3.pt"))
    b = make_agent(SPEC_B, os.path.join(CKPT, "teacher47_v3.pt"))
    refs = [RuleAgent("greedy_tenpai"), RuleAgent("call_first")]
    specs = [("hybrid", a), ("hybridd", b),
             ("greedy_tenpai", refs[0]), ("call_first", refs[1])]

    st = {name: {"games": 0, "wins": 0, "pts": 0.0, "dealin": 0}
          for name, _ in specs}
    t0 = time.time()
    for g in range(args.games):
        # 座位轮换：每个 agent 都轮流坐不同座位
        order = specs[g % 4:] + specs[:g % 4]
        agents = [x[1] for x in order]
        out = play_one(agents, seed=args.seed + g * 131)
        if out is None:
            continue
        delta, winner, _aux, _ad, dealin = out
        for pos, (name, _ag) in enumerate(order):
            s = st[name]
            s["games"] += 1
            s["pts"] += delta[pos]
            if winner == pos:
                s["wins"] += 1
            if dealin[pos]:
                s["dealin"] += 1
        if (g + 1) % 50 == 0:
            print(f"[danger] {g+1}/{args.games} ({time.time()-t0:.0f}s)",
                  flush=True)

    res = {}
    for name, s in st.items():
        n = max(1, s["games"])
        res[name] = {"games": s["games"],
                     "win_rate": round(100.0 * s["wins"] / n, 2),
                     "avg_pt": round(s["pts"] / n, 1),
                     "deal_in_rate": round(100.0 * s["dealin"] / n, 2)}

    d, lo, hi = two_prop_ci(st["hybrid"]["dealin"], st["hybrid"]["games"],
                             st["hybridd"]["dealin"], st["hybridd"]["games"])
    if hi < 0:
        verdict = "PASS（hybridd 放铳率显著下降）"
    elif lo > 0:
        verdict = "FAIL（hybridd 放铳率显著上升）"
    else:
        verdict = "INCONCLUSIVE（放铳率差异不显著）"

    lines = ["# 危险张验证：hybrid vs hybridd（2026-11-05）", "",
             f"> 同批 {args.games} 局、两版本同局占座、座位轮换；"
             "对手 = greedy_tenpai + call_first（都会立直/攻）。", "",
             "| Agent | 局数 | 和牌率 | 场均得分 | 放铳率 |",
             "|---|---|---|---|---|"]
    for name in ("hybrid", "hybridd", "greedy_tenpai", "call_first"):
        r = res[name]
        lines.append(f"| {name} | {r['games']} | {r['win_rate']}% | "
                     f"{r['avg_pt']:+.1f} | {r['deal_in_rate']}% |")
    lines += ["", "## 放铳率差异（hybridd − hybrid）", "",
              f"- 绝对差：{d * 100:+.2f} pt，95% CI [{lo * 100:+.2f}, "
              f"{hi * 100:+.2f}]", f"- 判定：**{verdict}**", "",
              "> 差值为负表示 hybridd 放铳更少。样本 "
              f"{st['hybrid']['games']} 局/版；若 INCONCLUSIVE 需继续加局。"]
    text = "\n".join(lines)
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    payload = {"games": args.games, "specs": [s[0] for s in specs],
               "results": res,
               "deal_in_diff": {"mean": round(d, 4), "ci_lo": round(lo, 4),
                                "ci_hi": round(hi, 4), "verdict": verdict}}
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print("\n" + text, flush=True)
    print(f"写入 {args.out_md}\n写入 {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())