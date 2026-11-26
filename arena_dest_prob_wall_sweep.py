"""dest_prob_v3 wall_hold_weight 扫描头对头：默认0.25 vs 0.0/0.15/0.4。

用法：python arena_dest_prob_wall_sweep.py --games 300 --out rules/H2H_dest_wall.md
"""
from __future__ import annotations
import os, sys, argparse, random, collections
sys.path.insert(0, os.getcwd())
import arena_elo as AE
from arena_elo import RuleAgent
import dest_prob_v3 as _DP
from mahjong import policies as POL

_DP.register_policy("dest_prob_v3")
for w in [0.0, 0.15, 0.4]:
    POL.POLICIES[f"dest_prob_v3_wh{int(w*100):02d}"] = (
        lambda seed=0, _w=w: _DP.DestProbV3(seed=seed, wall_hold_weight=_w))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--seed", type=int, default=4000)
    ap.add_argument("--out", default="rules/H2H_dest_wall.md")
    args = ap.parse_args()

    bots_def = [("dest_prob_v3", None), ("dest_prob_v3_wh00", None),
                ("dest_prob_v3_wh15", None), ("dest_prob_v3_wh40", None)]
    agents = [RuleAgent(nm, nm) for nm, _ in bots_def]

    n = len(agents)
    results = collections.Counter()
    pt_sum = collections.defaultdict(float)
    agari = collections.defaultdict(int)
    deal_in = collections.defaultdict(int)
    n_done = 0

    for g in range(args.games):
        seed = args.seed + g
        r = AE.play_one(agents, seed)
        if r is None:
            continue
        delta, winner, aux_bonus, aux_done, deal_in_flag, tenhou, scores = r
        order = sorted(range(n), key=lambda i: scores[i], reverse=True)
        for rank, pos in enumerate(order):
            bot = agents[pos].name
            results[(bot, rank)] += 1
            pt_sum[bot] += tenhou[pos]
        if winner >= 0:
            agari[agents[winner].name] += 1
        for pos in range(n):
            if deal_in_flag[pos]:
                deal_in[agents[pos].name] += 1
        n_done += 1
        if n_done % 25 == 0:
            print(f"[wall] {n_done}/{args.games}", flush=True)

    lines = [f"# dest_prob_v3 wall_hold_weight 扫描头对头（{n_done} 局，4 桌各一 bot）", "",
             "| bot | wall_hold | 平均pt | 1位% | 2位% | 3位% | 4位% | 和牌率 | 放铳率 |",
             "|---|---|---|---|---|---|---|---|---|"]
    walls = {"dest_prob_v3": "0.25(默认)", "dest_prob_v3_wh00": "0.0",
             "dest_prob_v3_wh15": "0.15", "dest_prob_v3_wh40": "0.40"}
    for nm, _ in bots_def:
        avg = pt_sum[nm] / n_done
        pcts = [results[(nm, r)] / n_done * 100 for r in range(4)]
        lines.append(f"| {nm} | {walls[nm]} | {avg:+.2f} | {pcts[0]:.1f}% | {pcts[1]:.1f}% | {pcts[2]:.1f}% | {pcts[3]:.1f}% | {agari[nm]/n_done*100:.1f}% | {deal_in[nm]/n_done*100:.1f}% |")
    lines += ["", f"- 总局数：{n_done}"]
    lines += [f"- {nm}({walls[nm]}) 平均 pt：{pt_sum[nm]/n_done:+.3f}" for nm, _ in bots_def]
    lines += ["", "> 注：4 个同类 bot 同桌互相克制，绝对数值不可与 vs pure_shanten 头对头对比；只看相对排序。"]
    out = "\n".join(lines)
    with open(args.out, "w") as f:
        f.write(out)
    print(out, flush=True)

if __name__ == "__main__":
    main()
