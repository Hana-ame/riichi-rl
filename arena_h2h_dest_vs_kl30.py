"""dest_prob_v3 vs kl01_it30 头对头（同桌 + 2 filler），判续训是否再提升。

用法：python arena_h2h_dest_vs_kl30.py --games 600 --out rules/H2H_dest_vs_kl30.md
"""
from __future__ import annotations
import os, sys, argparse, random, collections
sys.path.insert(0, os.getcwd())
import arena_elo as AE
from arena_elo import RuleAgent, HanchanAgent
import dest_prob_v3 as _DP
_DP.register_policy("dest_prob_v3")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--seed", type=int, default=13000)
    ap.add_argument("--out", default="rules/H2H_dest_vs_kl30.md")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    bots = [("dest_prob_v3", RuleAgent("dest_prob_v3", "dest_prob_v3")),
            ("kl01_it30", HanchanAgent("kl01_it30", os.path.join(AE.CKPT, "ppo_wuke_d2_kl01_it30_it15.pt"))),
            ("damaten", RuleAgent("damaten", "damaten_fill")),
            ("chi_only", RuleAgent("chi_only", "chi_only_fill"))]

    pt_sum = collections.defaultdict(float)
    results = collections.Counter()
    agari = collections.defaultdict(int)
    deal_in = collections.defaultdict(int)
    n_done = 0

    for g in range(args.games):
        order_bots = bots[:]
        rng.shuffle(order_bots)
        agents = [b[1] for b in order_bots]
        seat_bot = {i: order_bots[i][0] for i in range(4)}
        r = AE.play_one(agents, args.seed + g)
        if r is None:
            continue
        delta, winner, aux_bonus, aux_done, deal_in_flag, tenhou, scores = r
        order = sorted(range(4), key=lambda i: scores[i], reverse=True)
        ten = AE.tenhou_pt(scores)
        for rank, pos in enumerate(order):
            bot = seat_bot[pos]
            results[(bot, rank)] += 1
            pt_sum[bot] += ten[pos]
        if winner >= 0:
            agari[seat_bot[winner]] += 1
        for pos in range(4):
            if deal_in_flag[pos]:
                deal_in[seat_bot[pos]] += 1
        n_done += 1
        if n_done % 25 == 0:
            print(f"[h2h] {n_done}/{args.games}", flush=True)

    labels = {"dest_prob_v3": "dest_prob_v3(非ML)", "kl01_it30": "kl01_it30(高KL续训)",
              "damaten": "damaten(fill)", "chi_only": "chi_only(fill)"}
    lines = [f"# dest_prob_v3 vs kl01_it30 头对头（{n_done} 局，同桌各一）", "",
             "| bot | 平均pt | 1位% | 2位% | 3位% | 4位% | 和牌率 | 放铳率 |",
             "|---|---|---|---|---|---|---|---|"]
    for bot in ["dest_prob_v3", "kl01_it30", "damaten", "chi_only"]:
        avg = pt_sum[bot] / n_done
        pcts = [results[(bot, r)] / n_done * 100 for r in range(4)]
        lines.append(f"| {labels[bot]} | {avg:+.2f} | {pcts[0]:.1f}% | {pcts[1]:.1f}% | {pcts[2]:.1f}% | {pcts[3]:.1f}% | {agari[bot]/n_done*100:.1f}% | {deal_in[bot]/n_done*100:.1f}% |")
    diff = (pt_sum["dest_prob_v3"] - pt_sum["kl01_it30"]) / n_done
    lines += ["", f"- 总局数：{n_done}",
              f"- dest_prob_v3 平均 pt：{pt_sum['dest_prob_v3']/n_done:+.3f}",
              f"- kl01_it30 平均 pt：{pt_sum['kl01_it30']/n_done:+.3f}",
              f"- 差值(dest-kl30)：{diff:+.3f}"]
    out = "\n".join(lines)
    with open(args.out, "w") as f:
        f.write(out)
    print(out, flush=True)

if __name__ == "__main__":
    main()
