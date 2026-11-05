"""生成「固定槽位」BC 数据：用规则教师（GreedyTenpai）跑半庄，记录 47 槽标签。

背景（2026-11-05）：54 槽线 `train_reinforce.slot_of` 是哈希
`34+((type*7+tile)%20)`，非单射（实测 0.62% 决策点 CHI/PON 撞 PASS），
模型叫牌会被执行成 PASS、标签也被污染。半庄 47 槽线用**固定特殊槽**
（TSUMO=34/RON=35/RIICHI=36/…），无此问题。

本脚本按 47 槽固定映射录制 (obs 786 维 + touch 8 维, 合法槽 mask, 标签槽)，
供 `train_teacher_bc.py` 做带「特殊动作加权」的行为克隆。

用法：
    python gen_teacher_data.py --hanchans 12 --out data/teacher47.npz
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mahjong import env as E
from mahjong import obs2 as O2
from mahjong import policies as POL
from mahjong.hanchan import HanchanState
from train_hanchan import (encode, slot_of, hand_shanten, N_ACTIONS, SIT_DIM)


def run_hanchan(enc, h, teacher, seed0):
    """跑一个半庄，返回 (vecs, touches, masks, labels, slots_ok)。"""
    xs, tfs, masks, labels = [], [], [], []
    aux = []
    deals = 0
    while not h.finished:
        env = h.new_env(seed=seed0 + h.current_index() * 131 + deals)
        deal_ps = []
        r = env.start_turn()
        steps = 0
        while r is None and steps < 400:
            steps += 1
            acts = {}
            for p in env.pending_players():
                legal = env.legal_actions(p)
                if not legal:
                    acts[p] = E.PASS_ACTION
                    continue
                vec, sh, sit, gu, tf = encode(env, enc, h, p)
                slots = [slot_of(a) for a in legal]
                if any(s < 0 or s >= N_ACTIONS for s in slots):
                    acts[p] = POL.make_policy("greedy_tenpai").choose(
                        env, p, legal)
                    continue
                act = teacher.choose(env, p, legal)
                lab = slot_of(act)
                if lab not in slots:
                    # 教师动作必须是合法动作之一；否则退回第一个合法槽
                    lab = slots[0]
                mask = np.zeros(N_ACTIONS, dtype=bool)
                mask[slots] = True
                xs.append(vec)
                tfs.append(tf)
                masks.append(mask)
                labels.append(lab)
                deal_ps.append(p)
                acts[p] = act
            r = env.step(acts)
        if r is None:
            for _p0 in deal_ps:
                aux.append(0.0)
            break
        won = (r.result_type not in ("ryuukyoku",)) and getattr(r, "winner", None) is not None
        has_dora = (getattr(r, "dora_count", 0) + getattr(r, "aka_count", 0)
                    + getattr(r, "ura_dora_count", 0)) > 0
        for p0 in deal_ps:
            aux.append(1.0 if (won and r.winner == p0 and has_dora) else 0.0)
        deals += 1
        h.advance(r)
    return (np.stack(xs), np.stack(tfs).reshape(len(xs), -1),
            np.stack(masks), np.array(labels, dtype=np.int64),
            np.array(aux, dtype=np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hanchans", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--teacher", default="greedy_tenpai",
                    help="教师规则：greedy_tenpai/call_first/damaten/riichi_ev")
    ap.add_argument("--obs-order", action="store_true",
                    help="obs v3：加四家弃牌顺序平面（46 平面）")
    ap.add_argument("--out", default="data/teacher47.npz")
    args = ap.parse_args()

    enc = O2.ObsEncoder(order=getattr(args, "obs_order", False))
    teacher = POL.make_policy(getattr(args, "teacher", "greedy_tenpai"), seed=0)
    t0 = time.time()
    parts = []
    for i in range(args.hanchans):
        h = HanchanState(dealer=i % 4)
        parts.append(run_hanchan(enc, h, teacher, 10_000 + i * 7919))
        if (i + 1) % 3 == 0:
            print(f"  {i + 1}/{args.hanchans} 半庄 "
                  f"{sum(len(p[3]) for p in parts)} 样本 "
                  f"({time.time() - t0:.0f}s)", flush=True)
    X = np.concatenate([p[0] for p in parts])
    TF = np.concatenate([p[1] for p in parts])
    M = np.concatenate([p[2] for p in parts])
    Y = np.concatenate([p[3] for p in parts])
    AUX = np.concatenate([p[4] for p in parts])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, x=X, tf=TF, mask=M, y=Y, aux=AUX)
    print(f"样本 {len(X)}  观测 {X.shape[1]}  touch {TF.shape[1]}  "
          f"槽 {M.shape[1]}")
    for name, slot in (("tsumo", 34), ("ron", 35), ("riichi", 36)):
        n = int((Y == slot).sum())
        print(f"  {name:>6}: {n} ({100.0 * n / len(Y):.2f}%)")
    print(f"写入 {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()