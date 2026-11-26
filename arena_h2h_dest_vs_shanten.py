"""pure_shanten vs dest_prob_v3 大样本头对头（每次 4 人局，两 bot 各占两席）。"""
from __future__ import annotations
import os, sys, argparse, random, collections
sys.path.insert(0, os.getcwd())
import arena_elo as AE
from arena_elo import RuleAgent
import dest_prob_v3 as _DP
_DP.register_policy("dest_prob_v3")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=800)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", default="rules/H2H_dest_vs_shanten.md")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    a1 = RuleAgent("dest_prob_v3", "dest_prob_v3")
    a2 = RuleAgent("pure_shanten", "pure_shanten")
    fillers = [RuleAgent("random", "random_fill"), RuleAgent("chi_only", "chi_only_fill")]

    results = collections.Counter()  # (bot_name, rank) -> count
    pt_sum = {"dest_prob_v3": 0.0, "pure_shanten": 0.0}
    wins = {"dest_prob_v3": 0, "pure_shanten": 0}
    deal_in = {"dest_prob_v3": 0, "pure_shanten": 0}
    agari = {"dest_prob_v3": 0, "pure_shanten": 0}
    n_done = 0

    for g in range(args.games):
        # 随机分配座位：dest_prob_v3 和 pure_shanten 各占一个座位，2 个 filler
        bots = [("dest_prob_v3", a1), ("pure_shanten", a2), ("random", fillers[0]), ("chi_only", fillers[1])]
        rng.shuffle(bots)
        agents = [b[1] for b in bots]
        seat_bot = {i: bots[i][0] for i in range(4)}
        seed = args.seed + g
        r = AE.play_one(agents, seed)
        if r is None:
            continue
        delta, winner, aux_bonus, aux_done, deal_in_flag, tenhou, scores = r
        order = sorted(range(4), key=lambda i: scores[i], reverse=True)
        tenhou = AE.tenhou_pt(scores)
        for rank, pos in enumerate(order):
            bot = seat_bot[pos]
            if bot in pt_sum:
                results[(bot, rank)] += 1
                pt_sum[bot] += tenhou[pos]
        # 和牌 / 放铳：deal_in 是长度 4 的 bool 列表
        if winner >= 0 and seat_bot[winner] in pt_sum:
            agari[seat_bot[winner]] += 1
        for pos in range(4):
            bot = seat_bot[pos]
            if bot in pt_sum and deal_in_flag[pos]:
                deal_in[bot] += 1
        n_done += 1
        if n_done % 25 == 0:
            print(f"[h2h] {n_done}/{args.games} 局", flush=True)

    # 输出
    lines = ["# pure_shanten vs dest_prob_v3 头对头（%d 局）" % n_done, "",
             "| bot | 平均pt | 1位% | 2位% | 3位% | 4位% | 和牌率 | 放铳率 |", "|---|---|---|---|---|---|---|---|"]
    for bot in ["dest_prob_v3", "pure_shanten"]:
        avg_pt = pt_sum[bot] / n_done
        rank_pcts = [results[(bot, r)] / n_done * 100 for r in range(4)]
        lines.append(f"| {bot} | {avg_pt:+.2f} | {rank_pcts[0]:.1f}% | {rank_pcts[1]:.1f}% | {rank_pcts[2]:.1f}% | {rank_pcts[3]:.1f}% | {agari[bot]/n_done*100:.1f}% | {deal_in[bot]/n_done*100:.1f}% |")
    lines += ["", f"- 总局数：{n_done}", f"- dest_prob_v3 平均 pt：{pt_sum['dest_prob_v3']/n_done:+.3f}", f"- pure_shanten 平均 pt：{pt_sum['pure_shanten']/n_done:+.3f}", f"- 差值：{(pt_sum['dest_prob_v3']-pt_sum['pure_shanten'])/n_done:+.3f}"]
    out = "\n".join(lines)
    with open(args.out, "w") as f:
        f.write(out)
    print(out, flush=True)

if __name__ == "__main__":
    main()
