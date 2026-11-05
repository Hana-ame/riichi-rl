"""正式对战记录：把外部 Suphx 模型纳入 session 级 Elo 锦标赛。

4 参赛者：teacher47_100h_v2 / teacher47_v3 / Suphx 复现（291ch 适配）/ greedy。
每场按最终点数排名做 4 人两两 Elo（K=24，初值 1500），座位轮换。
用法：python external_elo.py --sessions 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cpu_only  # noqa
from arena_elo import RuleAgent, CKPT
from cpu_only import set_inference_threads
from mahjong import env as E
from mahjong.hanchan import HanchanState
from version_benchmark import make_agent

SPECS = ["teacher47_100h_v2", "teacher47_v3", "suphx", "greedy_tenpai"]


def load(spec):
    if spec in ("greedy_tenpai", "call_first"):
        return RuleAgent(spec)
    if spec == "suphx":
        return make_agent("suphx")
    return make_agent(spec, os.path.join(CKPT, spec + ".pt"))


def elo_update(r, order, K=24):
    for i in range(4):
        for j in range(i + 1, 4):
            a, b = order[i], order[j]
            sa = 1.0 if i < j else 0.0
            ea = 1.0 / (1.0 + 10 ** ((r[b] - r[a]) / 400.0))
            r[a] += K * (sa - ea)
            r[b] -= K * (sa - ea)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=12)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-md", default=os.path.join("rules", "EXTERNAL_ELO_2026-11-05.md"))
    ap.add_argument("--out-json", default=os.path.join("rules", "EXTERNAL_ELO_2026-11-05.json"))
    args = ap.parse_args()
    set_inference_threads(2)
    rating = {s: 1500.0 for s in SPECS}
    pts = {s: 0.0 for s in SPECS}
    wins = {s: 0 for s in SPECS}
    deals = {s: 0 for s in SPECS}
    for s in range(args.sessions):
        shift = s % 4
        names = SPECS[shift:] + SPECS[:shift]
        seats = [load(n) for n in names]
        h = HanchanState(dealer=s % 4)
        env = h.new_env(seed=args.seed + s * 977)
        for a in seats:
            a.set_game(env, h)
        while True:
            r = env.start_turn()
            while r is None:
                acts = {}
                for p in env.pending_players():
                    legal = env.legal_actions(p)
                    acts[p] = seats[p].act(env, p, legal) if legal else E.PASS_ACTION
                r = env.step(acts)
            for i in range(4):
                pts[names[i]] += r.points_delta[i]
            h.advance(r)
            if h.finished:
                break
            env = h.new_env(seed=args.seed + s * 977 + h.kyoku)
            for a in seats:
                a.set_game(env, h)
        elo_update(rating, [names[i] for i in h.final_placement()])
        if r.result_type != "ryuukyoku" and r.winner is not None:
            wins[names[r.winner]] += 1
        print(f"[external-elo] {s+1}/{args.sessions}", flush=True)
    lines = ["# 外部模型纳入的 session 级 Elo（2026-11-05）", "",
             f"> {args.sessions} 场半庄、4 参赛者（100h_v2/v3/Suphx 复现/greedy），",
             "> 按最终点数排名两两 Elo（K=24，初值 1500），座位每场轮换。", "",
             "| 参赛者 | Elo | 场均得分/场 | 和牌次数 |",
             "|---|---|---|---|"]
    for n in sorted(SPECS, key=lambda x: -rating[x]):
        lines.append(f"| {n} | **{rating[n]:.1f}** | {pts[n]/args.sessions:+.1f} | {wins[n]} |")
    text = "\n".join(lines) + "\n"
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(text)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({n: {"elo": round(rating[n], 1),
                       "avg_pts": round(pts[n] / args.sessions, 1),
                       "wins": wins[n]} for n in SPECS}, f,
                  ensure_ascii=False, indent=1)
    print(text)
    print("写入", args.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
