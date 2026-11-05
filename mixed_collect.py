"""混合对手数据采集：学生池 = bigBC / 100h_v2 / bc_v1 / greedy / call_first 混战，
教师 = greedy_tenpai 打 47 槽标签（修「异质对手分布」）。

python mixed_collect.py --deals 300 --out data/mixed300.npz
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mahjong import env as E
from mahjong import obs2 as O2
from mahjong.hanchan import HanchanState
from mahjong import policies as POL
from train_hanchan import encode, slot_of, N_ACTIONS
from version_benchmark import make_agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=300)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="data/mixed300.npz")
    a = ap.parse_args()
    torch.set_num_threads(2)
    pool = [
        ("big", "teacher47_300h_big", "checkpoints/teacher47_300h_big.pt"),
        ("100h", "teacher47_100h_v2", "checkpoints/teacher47_100h_v2.pt"),
        ("bc", "bc_v1_last", "checkpoints/bc_v1_last.pt"),
        ("greedy", "greedy_tenpai", None),
        ("callfirst", "call_first", None),
    ]
    teacher = POL.make_policy("greedy_tenpai", seed=0)
    enc = O2.ObsEncoder()
    xs, tfs, masks, ys = [], [], [], []
    t0 = time.time()
    for g in range(a.deals):
        h = HanchanState(dealer=g % 4); env = h.new_env(seed=a.seed + g * 131)
        seats = []
        for p in range(4):
            name, label, ck = pool[(g + p * 2) % len(pool)]
            if ck:
                ag = make_agent(label, ck)
                ag.set_game(env, h)
                seats.append(("agent", ag))
            else:
                seats.append(("rule", POL.make_policy(label, seed=a.seed + g * 71 + p)))
        r = env.start_turn(); steps = 0
        while r is None and steps < 400:
            steps += 1
            acts = {}
            for p in env.pending_players():
                legal = env.legal_actions(p)
                if not legal: acts[p] = E.PASS_ACTION; continue
                slots = [slot_of(a) for a in legal]
                if all(0 <= s < N_ACTIONS for s in slots):
                    lab = slot_of(teacher.choose(env, p, legal))
                    if lab in slots:
                        vec, sh, sit, gu, tf = encode(env, enc, h, p)
                        mask = np.zeros(N_ACTIONS, dtype=bool); mask[slots] = True
                        xs.append(vec); tfs.append(tf); masks.append(mask); ys.append(lab)
                kind, ag = seats[p]
                act = ag.act(env, p, legal) if kind == "agent" else ag.choose(env, p, legal)
                if act not in legal: act = E.PASS_ACTION if E.PASS_ACTION in legal else legal[0]
                acts[p] = act
            r = env.step(acts)
        h.advance(r) if r is not None else None
        if (g + 1) % 50 == 0:
            print(f"  {g+1}/{a.deals} 样本 {len(xs)} ({time.time()-t0:.0f}s)", flush=True)
    X = np.stack(xs); TF = np.stack(tfs).reshape(len(xs), -1)
    M = np.stack(masks); Y = np.array(ys, dtype=np.int64)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, x=X, tf=TF, mask=M, y=Y)
    print(f"样本 {len(X)} 观测 {X.shape[1]} → {a.out}")


if __name__ == "__main__":
    raise SystemExit(main())
