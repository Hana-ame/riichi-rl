"""东风 / 半庄分别的 session 级 ELO（4 个模型混桌，按场次结算）。

arena_elo 是「单局」ELO；本脚本回答「分模式 ELO」：
- 半庄 session：E1–S4 打完整场；东风 session：E1–E4；
- 每场按最终点数排名做 4 人两两 Elo 更新（K=24，初始 1500，逐对 1/0.5/0）；
- 4 个不同 Agent 固定同桌、座位每场轮换。

用法：
    python mode_elo.py --sessions 30 --out rules/MODE_ELO_2026-11-05.json
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

SPECS = ["teacher47_100h_v2", "teacher47_v3", "bc_v1_last", "greedy_tenpai"]
NAMES = {"teacher47_100h_v2": "100h_v2", "teacher47_v3": "v3",
         "bc_v1_last": "bc_v1", "greedy_tenpai": "greedy"}


def load(spec):
    if spec in ("greedy_tenpai", "call_first"):
        return RuleAgent(spec)
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
    ap.add_argument("--sessions", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=os.path.join("rules", "MODE_ELO_2026-11-05.json"))
    args = ap.parse_args()
    set_inference_threads(2)
    res = {}
    for mode, east in (("hanchan", False), ("east", True)):
        rating = {n: 1500.0 for n in SPECS}
        pts = {n: 0.0 for n in SPECS}
        deals = {n: 0 for n in SPECS}
        agents = [load(s) for s in SPECS]
        for s in range(args.sessions):
            order = list(range(4))
            shift = s % 4
            seats = agents[shift:] + agents[:shift]
            names = SPECS[shift:] + SPECS[:shift]
            h = HanchanState(dealer=s % 4, east_only=east)
            env = h.new_env(seed=args.seed + s * 977)
            for a in seats:
                a.set_game(env, h)
            while True:
                r = env.start_turn()
                while r is None:
                    acts = {}
                    for p in env.pending_players():
                        legal = env.legal_actions(p)
                        acts[p] = (seats[p].act(env, p, legal) if legal
                                   else E.PASS_ACTION)
                    r = env.step(acts)
                for i in range(4):
                    pts[names[i]] += r.points_delta[i]
                h.advance(r)
                if h.finished:
                    break
                env = h.new_env(seed=args.seed + s * 977 + h.kyoku)
                for a in seats:
                    a.set_game(env, h)
            rank = [names[i] for i in h.final_placement()]
            elo_update(rating, rank)
            for i in range(4):
                deals[names[i]] += 1
        rows = {n: {"elo": round(rating[n], 1),
                    "sessions": args.sessions,
                    "avg_pts_per_session": round(pts[n] / args.sessions, 1)}
                for n in SPECS}
        res[mode] = rows
        print(f"== {mode} ({args.sessions} 场) ==")
        for n in sorted(SPECS, key=lambda x: -rating[x]):
            print(f"  {NAMES[n]:<8} Elo {rating[n]:7.1f}  场均 {pts[n]/args.sessions:+8.1f}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("写入", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
