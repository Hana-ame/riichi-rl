"""演化策略（ES）训练麻将模型：无梯度，不怕奖励稀疏。

每代流程：
  1. 以当前精英参数为圆心，加高斯噪声生成 pop 个个体
  2. 每个个体坐 0 位打 games 局（对手=规则基线），适应度=平均点数变化+和了加成
  3. 取 top-k 个体的参数均值作为下一代圆心；最优个体存盘

和 PPO 的区别：PPO 靠梯度（需要密集奖励塑形），ES 只靠对局得分，
奖励再稀疏也能选优——但样本效率低，靠多进程对局数量堆。

用法：
    /mnt/d/WorkPlace/my_rl_project/venv/bin/python es_train.py \
        --pop 16 --gen 100 --games 4 --workers 4 --init checkpoints/riichi_net.pt
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mahjong import env as E
from mahjong import obs2 as O2
from mahjong import policies as POL
from mahjong.net import MahjongNet, count_params
from train_reinforce import slot_of, N_ACTIONS

ROOT = os.path.dirname(os.path.abspath(__file__))


def vec_of(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().numpy().copy()


def load_init(path):
    m = MahjongNet()
    if path and os.path.exists(path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m.load_state_dict(ck["state_dict"])
        print(f"init from {path}", flush=True)
    return vec_of(m)


# ---- 个体评估（子进程跑，必须顶层函数） ----

def eval_individual(task):
    vec, games, seed, temp, opp = task
    model = MahjongNet()
    torch.nn.utils.vector_to_parameters(
        torch.from_numpy(vec).float(), model.parameters())
    model.eval()
    enc = O2.ObsEncoder()
    total_pt, wins, deals = 0.0, 0, 0
    for g in range(games):
        env = E.MahjongEnv(seed=seed + g, dealer=(seed + g) % 4)
        r = env.start_turn()
        steps = 0
        while r is None and steps < 400:
            steps += 1
            actions = {}
            for p in env.pending_players():
                legal = env.legal_actions(p)
                if not legal:
                    actions[p] = E.PASS_ACTION
                    continue
                if len(legal) == 1:
                    actions[p] = legal[0]
                    continue
                if p == 0:
                    planes, scalar = enc.encode(env, p)
                    xin = np.concatenate([planes.ravel(), scalar]).astype(np.float32)
                    slots = [slot_of(a) for a in legal]
                    with torch.no_grad():
                        logits = model(torch.from_numpy(xin).unsqueeze(0))[0].reshape(-1)
                    mask = torch.full((N_ACTIONS,), -1e9, dtype=torch.float32)
                    mask[slots] = 0.0
                    lg = (logits + mask).numpy()
                    if temp > 0:
                        pr = np.exp((lg[slots] - lg[slots].max()) / temp)
                        pr /= pr.sum()
                        ai = int(np.random.choice(len(slots), p=pr))
                    else:
                        ai = int(lg[slots].argmax())
                    actions[p] = legal[ai]
                else:
                    actions[p] = POL.make_policy(opp, seed=seed * 97 + p + g).choose(env, p, legal)
            r = env.step(actions)
        if r is not None:
            deals += 1
            total_pt += E.net_points(r, [25000] * 4)[0]
            if r.result_type != "ryuukyoku" and r.winner == 0:
                wins += 1
    n = max(1, deals)
    fitness = total_pt / n + wins * 5000.0   # 点数为主，和了一把额外加成
    return fitness, wins, deals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--gen", type=int, default=100)
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=0.02)
    ap.add_argument("--elite-frac", type=float, default=0.25)
    ap.add_argument("--temp", type=float, default=0.7, help="评估时出牌采样温度")
    ap.add_argument("--opp", default="greedy_tenpai")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--init", default="checkpoints/riichi_net.pt")
    ap.add_argument("--out", default="checkpoints/es_v1.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    elite = load_init(args.init if args.init != "none" else None)
    dim = len(elite)
    rng = np.random.default_rng(args.seed)
    k = max(1, int(args.pop * args.elite_frac))
    best_ever = (elite, -1e18)
    print(f"ES: dim={dim:,} pop={args.pop} top{k} sigma={args.sigma} "
          f"games={args.games} workers={args.workers}", flush=True)

    for gen in range(args.gen):
        # 种群 = elite 本身 + pop-1 个扰动个体
        vecs = [elite] + [elite + args.sigma * rng.standard_normal(dim).astype(np.float32)
                          for _ in range(args.pop - 1)]
        tasks = [(v, args.games, args.seed + gen * 10007 + i * 131, args.temp, args.opp)
                 for i, v in enumerate(vecs)]
        with mp.Pool(args.workers) as pool:
            res = pool.map(eval_individual, tasks)

        order = np.argsort([-r[0] for r in res])
        top = [vecs[i] for i in order[:k]]
        elite = np.mean(top, axis=0).astype(np.float32)
        bf, bw, bd = res[order[0]]
        mf = float(np.mean([r[0] for r in res]))
        print(f"gen {gen:>3}  best_fit={bf:>9.0f} (win {bw}/{bd})  "
              f"pop_mean={mf:>9.0f}  elite_fit={res[0][0]:>9.0f}", flush=True)
        if bf > best_ever[1]:
            best_ever = (vecs[order[0]].copy(), bf)
            m = MahjongNet()
            torch.nn.utils.vector_to_parameters(
                torch.from_numpy(best_ever[0]).float(), m.parameters())
            torch.save({"state_dict": m.state_dict(), "arch": "convformer_v1",
                        "trained_on": f"ES gen{gen} fit={bf:.0f}"}, args.out)

    print(f"done. best_fit={best_ever[1]:.0f} -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
