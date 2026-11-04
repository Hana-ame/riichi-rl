"""引擎健诊：N 局自对弈的规则级统计（不训模型，只验规则/策略读数）。

用途：
  1. 回答「修复后荣和（ron）到底偏少多少」——`_can_ron` 的有役检查是否过严；
  2. 每局验点数守恒（sum(scores)+kyokuten == 4*25000）与 delta 守恒；
  3. 记录听牌/立直/和牌/流局读数，做可证伪的回归基线。

跑法：
    /mnt/d/WorkPlace/my_rl_project/venv/bin/python engine_stats.py --games 200
    /mnt/d/WorkPlace/my_rl_project/venv/bin/python engine_stats.py --games 200 \\
        --out-json rules/ENGINE_STATS_2026-11-04.json

判据（写进报告，不做「看起来差不多」的判断）：
  - 点数守恒违规数必须为 0；
  - ron 结算数 > 0（修复「立直者被排除在响应阶段」前恒为 0）；
  - 「牌型成立但 _can_ron=False」分两类：furiten / 无役，分别计数。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mahjong import env as E
from mahjong import policies as POL


START_POINTS = 25000
N_PLAYERS = 4


def play_one(seed, policy_name="greedy_tenpai", policy_seed=0):
    """跑一局自对弈，返回统计 dict。"""
    env = E.MahjongEnv(seed=seed, dealer=seed % N_PLAYERS)
    start_total = sum(p.points for p in env.players)
    pols = {i: POL.make_policy(policy_name, seed=policy_seed * 97 + i)
            for i in range(N_PLAYERS)}
    st = {
        "seed": seed,
        "result_type": None,
        "winner": None,
        "from_player": None,
        "turns": 0,
        "riichi": 0,
        "tsumo_legal": 0,
        "tsumo_selected": 0,
        "tsumo_settled": 0,
        "ron_legal": 0,
        "ron_selected": 0,
        "ron_settled": 0,
        "shape_only": 0,        # 牌型可和但 _can_ron=False
        "shape_furiten": 0,     #   └ 其中因振听
        "shape_no_yaku": 0,     #   └ 其中因无役
        "tenpai_at_draw": 0,    # 摸牌后（弃牌阶段）自己处于听牌的次数
        "tenpai_reach": [False, False, False, False],  # 本局是否进过听牌
        "ryuukyoku_reason": None,
        "conservation_ok": None,
        "delta_ok": None,
        "start_kyokuten": 0,
        "scores": None,
        "kyokuten": 0,
    }
    result = env.start_turn()
    steps = 0
    while result is None and steps < 800:
        steps += 1
        acts = {}
        for p in env.pending_players():
            legal = env.legal_actions(p)
            if not legal:
                acts[p] = E.PASS_ACTION
                continue
            if not st["tenpai_reach"][p] and env._is_tenpai(p):
                st["tenpai_reach"][p] = True
            # --- 规则读数（先记，再让策略选）---
            if env.phase == E.Phase.DISCARD and p == env.turn_player:
                if env._is_tenpai(p):
                    st["tenpai_at_draw"] += 1
                if env._can_tsumo(p):
                    st["tsumo_legal"] += 1
            if env.phase == E.Phase.RESPONSE and env.last_discard is not None:
                tile = env.last_discard
                pp = env.players[p]
                from mahjong import refengine as REF
                shape = REF.can_win_with(pp.full_counts(), tile,
                                         len(pp.melds))
                if shape:
                    if env._can_ron(p):
                        st["ron_legal"] += 1
                    else:
                        st["shape_only"] += 1
                        if pp.furiten:
                            st["shape_furiten"] += 1
                        else:
                            st["shape_no_yaku"] += 1
            act = pols[p].choose(env, p, legal)
            if act.type == E.ACT_RIICHI:
                st["riichi"] += 1
            elif act.type == E.ACT_TSUMO:
                st["tsumo_selected"] += 1
            elif act.type == E.ACT_RON:
                st["ron_selected"] += 1
            acts[p] = act
        result = env.step(acts)

    if result is None:
        st["result_type"] = "unfinished"
        return st

    st["result_type"] = result.result_type
    st["winner"] = result.winner
    st["from_player"] = result.from_player
    st["turns"] = result.turns
    st["scores"] = list(result.scores)
    st["kyokuten"] = getattr(result, "kyokuten", 0)
    st["ryuukyoku_reason"] = getattr(result, "reason", "") or None
    if result.result_type == "tsumo":
        st["tsumo_settled"] = 1
    elif result.result_type == "ron":
        st["ron_settled"] = 1
    # 守恒：各家应收点数 + 桌上供托 == 开局总点数 + 起手供托
    st["start_kyokuten"] = env.initial_kyokuten
    st["tenpai_reach"] = [bool(x) for x in st["tenpai_reach"]]
    total = sum(result.scores) + st["kyokuten"]
    st["conservation_ok"] = (total == start_total + env.initial_kyokuten)
    # delta 守恒：sum(delta) = 本局立直供托 + 起手供托 - 结束供托
    # （立直扣款直接记在 players.points 里，不在 delta 里，故要加回）
    st["delta_ok"] = (sum(result.points_delta)
                      == st["riichi"] * 1000 + env.initial_kyokuten
                      - st["kyokuten"])
    return st


def aggregate(rows, elapsed):
    n = len(rows)
    ok = [r for r in rows if r["result_type"] not in (None, "unfinished")]
    out = {
        "games": n,
        "finished": len(ok),
        "elapsed_sec": round(elapsed, 2),
        "policy": "greedy_tenpai x4 (self-play)",
        "results": {},
        "rates_per_game": {},
        "totals": {},
        "violations": {},
    }
    for key in ("tsumo", "ron", "ryuukyoku", "unfinished"):
        c = sum(1 for r in rows if r["result_type"] == key)
        out["results"][key] = c
        out["rates_per_game"][key] = round(c / n, 4) if n else None
    if ok:
        m = max(r["turns"] for r in ok)
        out["totals"]["turns_max"] = m
        out["totals"]["turns_mean"] = round(
            sum(r["turns"] for r in ok) / len(ok), 2)
        for key in ("riichi", "tenpai_at_draw", "tsumo_legal",
                    "ron_legal", "shape_only", "shape_furiten",
                    "shape_no_yaku"):
            s = sum(r[key] for r in ok)
            out["totals"][key] = s
            out["rates_per_game"][key] = round(s / len(ok), 2)
    out["violations"]["conservation"] = sum(
        1 for r in rows if r["conservation_ok"] is False)
    out["violations"]["delta"] = sum(1 for r in rows if r["delta_ok"] is False)
    out["violations"]["unfinished"] = out["results"]["unfinished"]
    out["tenpai"] = {
        "seats_reached": sum(1 for r in ok for x in r["tenpai_reach"] if x),
        "seat_deals": 4 * len(ok),
        "seat_rate_pct": round(100.0 * sum(
            1 for r in ok for x in r["tenpai_reach"] if x)
            / max(1, 4 * len(ok)), 1),
        "deals_with_any_tenpai": sum(
            1 for r in ok if any(r["tenpai_reach"])),
        "deal_rate_pct": round(100.0 * sum(
            1 for r in ok if any(r["tenpai_reach"])) / max(1, len(ok)), 1),
    }
    out["totals"]["ron_legal_to_settled"] = (
        f'{out["totals"]["ron_legal"]}'
        f' -> {out["results"].get("ron", 0)} settled'
        if ok else None)
    out["totals"]["tsumo_legal_to_settled"] = (
        f'{out["totals"]["tsumo_legal"]}'
        f' -> {out["results"].get("tsumo", 0)} settled'
        if ok else None)
    return out


def main():
    ap = argparse.ArgumentParser(description="N 局自对弈的规则级读数")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed-base", type=int, default=10000)
    ap.add_argument("--out-json", default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    rows = []
    for k in range(args.games):
        rows.append(play_one(args.seed_base + k))
        if not args.quiet and (k + 1) % 50 == 0:
            print(f"  {k + 1}/{args.games} 局 ...", flush=True)
    elapsed = time.time() - t0
    agg = aggregate(rows, elapsed)

    print(json.dumps(agg, ensure_ascii=False, indent=2))
    if args.out_json:
        payload = {"aggregate": agg, "per_game": rows}
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        print(f"\n写入 {args.out_json}")
    return 0 if agg["violations"]["conservation"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())