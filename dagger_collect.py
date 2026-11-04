"""DAgger 数据采集：学生自对弈，教师在学生到达的状态上打标。

学生：任意 47 槽 ckpt（auto 载入）；教师：GreedyTenpai（规则）。
每局 4 席都是学生（self-play），但每个决策点额外记录教师的合法动作选择，
形成 (obs786+touch8, legal mask, 教师槽标签)，格式与 gen_teacher_data.npz 一致，
可直接喂 train_teacher_bc.py。

用法：
    python dagger_collect.py --ckpt checkpoints/teacher47_100h_v2.pt \
        --deals 120 --out data/dagger47_v1.npz
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from arena_elo import CKPT
from mahjong import env as E
from mahjong import obs2 as O2
from mahjong import policies as POL
from mahjong.hanchan import HanchanState
from train_hanchan import encode, slot_of, N_ACTIONS, SIT_DIM
from version_benchmark import AutoAgent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(
        CKPT, "teacher47_100h_v2.pt"))
    ap.add_argument("--deals", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--out", default=os.path.join(
        ROOT := os.path.dirname(os.path.abspath(__file__)),
        "data", "dagger47_v1.npz"))
    args = ap.parse_args()

    torch.set_num_threads(2)
    students = [AutoAgent("student", args.ckpt) for _ in range(4)]
    teacher = POL.make_policy("greedy_tenpai", seed=0)
    enc = O2.ObsEncoder()
    xs, tfs, masks, ys = [], [], [], []
    t0 = time.time()
    for g in range(args.deals):
        h = HanchanState(dealer=(args.seed + g * 131) % 4)
        env = h.new_env(seed=args.seed + g * 131)
        for a in students:
            a.set_game(env, h)
        r = env.start_turn()
        steps = 0
        while r is None and steps < 500:
            steps += 1
            acts = {}
            for p in env.pending_players():
                legal = env.legal_actions(p)
                if not legal:
                    acts[p] = E.PASS_ACTION
                    continue
                # 教师标签
                t_act = teacher.choose(env, p, legal)
                slots = [slot_of(a) for a in legal]
                if all(0 <= s < N_ACTIONS for s in slots):
                    lab = slot_of(t_act)
                    if lab in slots:
                        vec, sh, sit, gu, tf = encode(env, enc, h, p)
                        mask = np.zeros(N_ACTIONS, dtype=bool)
                        mask[slots] = True
                        xs.append(vec); tfs.append(tf)
                        masks.append(mask); ys.append(lab)
                acts[p] = students[p].act(env, p, legal)
            r = env.step(acts)
        if (g + 1) % 20 == 0:
            print(f"  {g+1}/{args.deals} 局 样本 {len(xs)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    X = np.stack(xs); TF = np.stack(tfs).reshape(len(xs), -1)
    M = np.stack(masks); Y = np.array(ys, dtype=np.int64)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, x=X, tf=TF, mask=M, y=Y)
    print(f"样本 {len(X)} 观测 {X.shape[1]} touch {TF.shape[1]} → {args.out}")
    for name, slot in (("tsumo", 34), ("ron", 35), ("riichi", 36)):
        n = int((Y == slot).sum())
        print(f"  {name}: {n} ({100.0*n/len(Y):.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
