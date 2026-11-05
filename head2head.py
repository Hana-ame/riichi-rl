"""头对头：自己模型 vs 别的模型（同 4 人桌、两席固定、座位轮换）。

每对打 N 局：A/B 各占一席并轮换座位，另外两席用 greedy_tenpai 规则（保底速度快、
会立直会进攻），统计 A 与 B 各自的：
和牌局数、放铳局数、平均点数（净分）、顺位分布。比较只用「同桌同池」数据。

用法：
    python head2head.py --agent teacher47_100h_v2 --games 60 \
        --opponents "teacher47_v3,bc_v1_last,hybridd:teacher47_v3,from_games_v2,greedy_tenpai"
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
from mahjong.hanchan import HanchanState
from mahjong import env as E
from version_benchmark import make_agent


def build(spec):
    if spec in ("greedy_tenpai", "call_first"):
        return RuleAgent(spec)
    base = spec.split(":", 1)[1] if ":" in spec else spec
    path = os.path.join(CKPT, base if base.endswith(".pt") else base + ".pt")
    return make_agent(spec, path)


def play_pair(a_spec, b_spec, games, seed):
    A, B = build(a_spec), build(b_spec)
    refs = [RuleAgent("greedy_tenpai"), RuleAgent("greedy_tenpai")]
    st = {"a": {"win": 0, "dealin": 0, "pts": 0.0, "place": [0, 0, 0, 0]},
          "b": {"win": 0, "dealin": 0, "pts": 0.0, "place": [0, 0, 0, 0]},
          "games": 0, "draw": 0}
    for g in range(games):
        sa = g % 4
        sb = (g // 4 + 1) % 4
        if sb == sa:
            sb = (sb + 1) % 4
        seats = [None] * 4
        seats[sa] = A
        seats[sb] = B
        ref_i = 0
        for i in range(4):
            if seats[i] is None:
                seats[i] = refs[ref_i]
                ref_i += 1
        h = HanchanState(dealer=g % 4)
        env = h.new_env(seed=seed + g * 131)
        for x in seats:
            x.set_game(env, h)
        r = env.start_turn()
        while r is None:
            acts = {}
            for p in env.pending_players():
                legal = env.legal_actions(p)
                acts[p] = seats[p].act(env, p, legal) if legal else E.PASS_ACTION
            r = env.step(acts)
        st["games"] += 1
        if r.result_type == "ryuukyoku":
            st["draw"] += 1
        else:
            if r.winner == sa:
                st["a"]["win"] += 1
            elif r.winner == sb:
                st["b"]["win"] += 1
            if r.result_type == "ron":
                if r.from_player == sa:
                    st["a"]["dealin"] += 1
                elif r.from_player == sb:
                    st["b"]["dealin"] += 1
        st["a"]["pts"] += r.points_delta[sa]
        st["b"]["pts"] += r.points_delta[sb]
        order = sorted(range(4), key=lambda i: (-r.scores[i], i))
        st["a"]["place"][order.index(sa)] += 1
        st["b"]["place"][order.index(sb)] += 1
    n = max(1, st["games"])
    for k in ("a", "b"):
        s = st[k]
        s["win_rate"] = round(100.0 * s["win"] / n, 2)
        s["dealin_rate"] = round(100.0 * s["dealin"] / n, 2)
        s["avg_pts"] = round(s["pts"] / n, 1)
        s["top1_rate"] = round(100.0 * s["place"][0] / n, 1)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="teacher47_100h_v2")
    ap.add_argument("--opponents", default=(
        "teacher47_v3,bc_v1_last,hybridd:teacher47_v3,from_games_v2,greedy_tenpai"))
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--out-json", default=os.path.join(
        "rules", "HEAD2HEAD_2026-11-05.json"))
    ap.add_argument("--out-md", default=os.path.join(
        "rules", "HEAD2HEAD_2026-11-05.md"))
    args = ap.parse_args()
    set_inference_threads(2)
    rows = {}
    for opp in [x.strip() for x in args.opponents.split(",") if x.strip()]:
        st = play_pair(args.agent, opp, args.games, args.seed)
        rows[opp] = st
        a, b = st["a"], st["b"]
        print(f"{args.agent:<20} vs {opp:<22} 局{st['games']:>3} | "
              f"和牌 {a['win_rate']:>5.2f}% vs {b['win_rate']:>5.2f}% | "
              f"放铳 {a['dealin_rate']:>5.2f}% vs {b['dealin_rate']:>5.2f}% | "
              f"场均 {a['avg_pts']:>+7.1f} vs {b['avg_pts']:>+7.1f} | "
              f"一位率 {a['top1_rate']:>4.1f}% vs {b['top1_rate']:>4.1f}%", flush=True)
    lines = [f"# 头对头：{args.agent} vs 别的模型（{args.games} 局/对，同池）", "",
             "> 每局 A/B 各占一席、座位轮换，另两席 greedy_tenpai；统计同桌同池。", "",
             "| 对手 | 局数 | A 和牌率 | B 和牌率 | A 放铳率 | B 放铳率 | A 场均 | B 场均 | A 一位率 | B 一位率 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for opp, st in rows.items():
        a, b = st["a"], st["b"]
        lines.append(f"| {opp} | {st['games']} | {a['win_rate']}% | {b['win_rate']}% | "
                     f"{a['dealin_rate']}% | {b['dealin_rate']}% | {a['avg_pts']:+.1f} | "
                     f"{b['avg_pts']:+.1f} | {a['top1_rate']}% | {b['top1_rate']}% |")
    os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"agent": args.agent, "games": args.games, "rows": rows}, f,
                  ensure_ascii=False, indent=1)
    print("写入", args.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
