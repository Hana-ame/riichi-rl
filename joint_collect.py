"""联合训练数据：学生 = 主模型 + 学习叫牌（CallAgent） roll-in；教师 = SuphxFullAgent。

在「学生会叫牌」的真实状态分布上，由强外部教师给 47 槽动作标签（含弃牌/吃碰杠），
用于「叫牌 + 后续打牌」联合 BC/DAgger。

用法：python joint_collect.py --deals 200 --out data/joint_v1.npz
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arena_elo import CKPT
from mahjong import env as E
from mahjong import obs2 as O2
from mahjong.hanchan import HanchanState
from train_hanchan import encode, slot_of, N_ACTIONS, SIT_DIM
from call_strategy import CallAgent
from call_teacher_suphx import SuphxFullAgent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=os.path.join("data", "joint_v1.npz"))
    args = ap.parse_args()
    torch.set_num_threads(2)
    students = [CallAgent("teacher47_100h_v2", os.path.join(CKPT, "teacher47_100h_v2.pt"),
                          os.path.join(CKPT, "call_policy_v2.pt")) for _ in range(4)]
    teacher = SuphxFullAgent("suphxfull")
    enc = O2.ObsEncoder()
    xs, tfs, masks, ys = [], [], [], []
    t0 = time.time()
    for g in range(args.deals):
        h = HanchanState(dealer=g % 4)
        env = h.new_env(seed=args.seed + g * 131)
        for a in students:
            a.set_game(env, h)
        teacher.set_game(env, h)
        r = env.start_turn(); steps = 0
        while r is None and steps < 500:
            steps += 1
            acts = {}
            for p in env.pending_players():
                legal = env.legal_actions(p)
                if not legal:
                    acts[p] = E.PASS_ACTION
                    continue
                t_act = teacher.act(env, p, legal)
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
            print(f"  {g+1}/{args.deals} 局 样本 {len(xs)} ({time.time()-t0:.0f}s)", flush=True)
    X = np.stack(xs); TF = np.stack(tfs).reshape(len(xs), -1)
    M = np.stack(masks); Y = np.array(ys, dtype=np.int64)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, x=X, tf=TF, mask=M, y=Y)
    print(f"样本 {len(X)} 观测 {X.shape[1]} touch {TF.shape[1]} → {args.out}")
    for name, slot in (("tsumo", 34), ("ron", 35), ("riichi", 36),
                       ("pon", 39), ("chi", 44)):
        n = int((Y == slot).sum())
        print(f"  {name}: {n} ({100.0*n/len(Y):.2f}%)")


if __name__ == "__main__":
    raise SystemExit(main())
