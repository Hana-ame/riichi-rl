"""固定参赛版 arena：所有 bot 每局都在场（无选员淘汰），消除选员随机性。

用于测 dest_prob_v3 的可靠 Elo（vs eff_wuke_d2 / teacher47_v3 / pure_shanten 等）。

用法：python arena_fixed_elo.py --games 600 --out rules/ELO_fixed.md
"""
from __future__ import annotations
import os, sys, argparse, collections, math
sys.path.insert(0, os.getcwd())
import arena_elo as AE
from arena_elo import RuleAgent, HanchanAgent
import dest_prob_v3 as _DP
_DP.register_policy("dest_prob_v3")

CKPT = AE.CKPT

def build():
    P = []
    P.append(RuleAgent("dest_prob_v3", "dest_prob_v3(非ML)"))
    P.append(RuleAgent("pure_shanten", "pure_shanten(纯向听)"))
    P.append(RuleAgent("riichi_needed", "riichi_needed(没役才立直)"))
    P.append(RuleAgent("riichi_ev", "riichi_ev(按点数期望)"))
    P.append(RuleAgent("damaten", "damaten(默听)"))
    # ML
    for nm, fn in [("eff_wuke_d2(牌效d2)", "teacher47_eff_wuke_d2.pt"),
                   ("ppo_wuke_d2_it30(PPO)", "ppo_wuke_d2_it30.pt"),
                   ("ppo_wuke_d2_it20(PPO)", "ppo_wuke_d2_it20.pt"),
                   ("teacher47_v3(加权BC)", "teacher47_v3.pt"),
                   ("teacher47_100h_v2(最优BC)", "teacher47_100h_v2.pt"),
                   ("tenhou_bc(Tenhou人类)", "tenhou_bc.pt")]:
        p = os.path.join(CKPT, fn)
        if os.path.exists(p):
            try:
                P.append(HanchanAgent(nm, p))
            except Exception as e:
                print(f"[skip] {nm}: {e}", flush=True)
    return P

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--out", default="rules/ELO_fixed.md")
    args = ap.parse_args()

    agents = build()
    n = len(agents)
    names = [a.name for a in agents]
    print(f"[fixed] {n} 个参赛者固定常驻: {names}", flush=True)

    K = 24.0
    ratings = {i: 1500.0 for i in range(n)}
    results = collections.Counter()
    pt_sum = collections.defaultdict(float)
    agari = collections.defaultdict(int)
    deal_in = collections.defaultdict(int)
    n_done = 0

    for g in range(args.games):
        # 每局随机抽 4 个参赛者上桌（不放回），保证每人局数均衡
        import random
        rng = random.Random(args.seed + g)
        seats = rng.sample(range(n), 4)
        table = [agents[i] for i in seats]
        r = AE.play_one(table, args.seed + g)
        if r is None:
            continue
        delta, winner, aux_bonus, aux_done, deal_in_flag, tenhou, scores = r
        order = sorted(range(4), key=lambda p: scores[p], reverse=True)
        # 逐对零和 Elo 更新（按最终点数排名）
        for a_idx in range(4):
            for b_idx in range(a_idx + 1, 4):
                ia, ib = seats[order[a_idx]], seats[order[b_idx]]
                ea = 1.0 / (1.0 + 10 ** ((ratings[ib] - ratings[ia]) / 400.0))
                sa = 1.0 if a_idx < b_idx else 0.0
                # 顺位差越大更新越大
                gap = (b_idx - a_idx)
                d = K * gap * (sa - ea)
                ratings[ia] += d
                ratings[ib] -= d
        for rank, pos in enumerate(order):
            i = seats[pos]
            results[(i, rank)] += 1
            pt_sum[i] += tenhou[pos]
        if winner >= 0:
            agari[seats[winner]] += 1
        for pos in range(4):
            if deal_in_flag[pos]:
                deal_in[seats[pos]] += 1
        n_done += 1
        if n_done % 25 == 0:
            print(f"[fixed] {n_done}/{args.games}", flush=True)

    # 输出
    lines = [f"# 固定参赛 Elo 排名（{n_done} 局，{n} 人，每局随机抽 4 人）", "",
             "> 无选员淘汰，每人局数均衡，消除选员随机性。逐对零和 Elo（K=24）。", "",
             "| 排名 | 参赛者 | 类型 | Elo | 局数 | 1位% | 2位% | 3位% | 4位% | 和牌率 | 放铳率 | 平均天凤pt |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    ranked = sorted(range(n), key=lambda i: ratings[i], reverse=True)
    for rank, i in enumerate(ranked, 1):
        nm = names[i]
        is_rule = isinstance(agents[i], RuleAgent)
        typ = "rule" if is_rule else "ML"
        gp = sum(results[(i, r)] for r in range(4))
        pcts = [results[(i, r)] / gp * 100 for r in range(4)]
        avg_pt = pt_sum[i] / gp
        lines.append(f"| {rank} | {nm} | {typ} | **{ratings[i]:.1f}** | {gp} | {pcts[0]:.1f}% | {pcts[1]:.1f}% | {pcts[2]:.1f}% | {pcts[3]:.1f}% | {agari[i]/gp*100:.1f}% | {deal_in[i]/gp*100:.1f}% | {avg_pt:+.2f} |")
    lines += ["", "> 判读：每人局数均衡；Elo 差 <50 分不宜当结论（K=24 收敛慢）。"]
    out = "\n".join(lines)
    with open(args.out, "w") as f:
        f.write(out)
    print(out, flush=True)

if __name__ == "__main__":
    main()
