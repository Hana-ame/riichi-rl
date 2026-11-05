"""实战对局直播：让不同模型真打半庄，逐局打印每家打了什么牌。

用法：
    python show_match.py --agents "teacher47_100h_v2,teacher47_v3,hybridd:teacher47_v3,greedy_tenpai"
    python show_match.py --hanchans 3 --verbose   # 还要逐手动作（含 PASS）
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cpu_only  # noqa
from arena_elo import CKPT
from mahjong import env as E
from mahjong import tiles as T
from mahjong.hanchan import HanchanState
from version_benchmark import make_agent

NAMES = {"teacher47_v1": "v1", "teacher47_v2": "v2", "teacher47_v3": "v3",
         "teacher47_100h": "100h", "teacher47_100h_v2": "100h_v2",
         "teacher47_dagger_mix_v2": "dg_mix2", "greedy_tenpai": "greedy"}


def load(spec):
    base = spec.split(":", 1)[1] if ":" in spec else spec
    path = os.path.join(CKPT, base if base.endswith(".pt") else base + ".pt")
    return make_agent(spec, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", default=(
        "teacher47_100h_v2,teacher47_v3,hybridd:teacher47_v3,greedy_tenpai"))
    ap.add_argument("--hanchans", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--verbose", action="store_true", help="逐手打印（含 PASS）")
    args = ap.parse_args()

    specs = [s.strip() for s in args.agents.split(",")]
    agents = [load(s) for s in specs]
    short = [NAMES.get(s.split(":", 1)[-1], s.split(":", 1)[-1][:8]) for s in specs]
    for g in range(args.hanchans):
        h = HanchanState(dealer=g % 4)
        env = h.new_env(seed=args.seed + g * 997)
        for a in agents:
            a.set_game(env, h)
        print(f"\n===== 第 {g+1} 半庄：{' | '.join(short)} =====")
        while not h.finished:
            seq = {p: [] for p in range(4)}
            r = env.start_turn()
            while r is None:
                acts = {}
                for p in env.pending_players():
                    legal = env.legal_actions(p)
                    if not legal:
                        acts[p] = E.PASS_ACTION
                        continue
                    a = agents[p].act(env, p, legal)
                    acts[p] = a
                    if a.type == E.ACT_PASS and not args.verbose:
                        continue
                    if a.type == E.ACT_DISCARD:
                        tile = a.tile if a.tile < 34 else a.tile // 4
                        seq[p].append(T.tile_name(tile))
                    elif a.type == E.ACT_CHI:
                        seq[p].append(f"[吃{T.tile_name(a.tile) if a.tile<34 else T.tile_name(a.tile//4)}]")
                    else:
                        seq[p].append("[" + E.ACT_NAMES[a.type] + "]")
                r = env.step(acts)
            tag = {"ron": "荣和", "tsumo": "自摸", "ryuukyoku": "流局"}
            res = (f"{tag.get(r.result_type, r.result_type)} "
                   f"{(short[r.winner] if r.winner is not None else '')}"
                   + (f" 放铳 {short[r.from_player]}" if r.result_type == "ron" else "")
                   + (f" {r.han or 0}番{r.fu or 0}符" if r.result_type != "ryuukyoku" else "")
                   + f" delta={list(r.points_delta)}")
            print(f"{h.round}{h.kyoku}局 {res}")
            for p in range(4):
                if seq[p]:
                    print(f"   {short[p]:<8}: " + " ".join(seq[p]))
            h.advance(r)
            if not h.finished:
                round_idx = 0 if h.round == "E" else 4
                env = h.new_env(seed=args.seed + g * 997 + (round_idx + h.kyoku) * 37)
                for a in agents:
                    a.set_game(env, h)
        order = h.final_placement()
        print("最终：" + "  ".join(f"{short[i]}={h.sticks[i]}" for i in order))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
