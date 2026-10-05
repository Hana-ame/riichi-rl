
from __future__ import annotations

import argparse
import glob
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mahjong import env as E
from mahjong import obs2 as O2
from mahjong import touch as TOUCH
from mahjong.net import MahjongNet
from mahjong.hanchan import HanchanState
from train_hanchan import N_ACTIONS, SIT_DIM, encode, slot_of, hand_shanten


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = MahjongNet(channels=ck.get("channels", 64),
                  n_blocks=ck.get("blocks", 4),
                  n_actions=N_ACTIONS,
                  touch_dim=ck.get("touch_dim", TOUCH.TOUCH_DIM),
                  extra_scalar=ck.get("sit_dim", SIT_DIM))
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


def run(model, enc, deals, temp, seed0):
    """纯随机采样（不用教师），学生自己对打。"""
    wins = 0
    d = 0
    sh_sum = 0
    sh_n = 0
    for i in range(deals):
        h = HanchanState(dealer=i % 4)
        while not h.finished:
            e = h.new_env(seed=seed0 + i * 13 + h.current_index())
            r = e.start_turn()
            g = 0
            while r is None and g < 400:
                g += 1
                pend = {}
                idxs = []
                acts = {}
                for p in e.pending_players():
                    vec, sh, sit, gu, tf = encode(e, enc, h, p)
                    sh_sum += max(0, sh)
                    sh_n += 1
                    legal = e.legal_actions(p)
                    if not legal:
                        pend[p] = None
                        continue
                    slots = [slot_of(a) for a in legal]
                    pend[p] = (vec, tf, slots, legal)
                    idxs.append(p)
                if idxs:
                    x = torch.from_numpy(np.stack([pend[p][0] for p in idxs]))
                    tx = torch.from_numpy(np.stack([pend[p][1] for p in idxs]))
                    with torch.no_grad():
                        lg, _ = model(x, tx)
                    for k, p in enumerate(idxs):
                        _, _, slots, legal = pend[p]
                        m = torch.full((N_ACTIONS,), -1e9)
                        m[slots] = 0.0
                        q = lg[k] + m
                        if temp <= 0:
                            pk = int(q.argmax())
                        else:
                            pk = int(torch.multinomial(
                                torch.softmax(q / temp, -1), 1))
                        if pk not in slots:
                            pk = slots[0]
                        acts[p] = legal[slots.index(pk)]
                for p in pend:
                    if pend[p] is None:
                        acts[p] = E.PASS_ACTION
                r = e.step(acts)
            if r is None:
                break
            d += 1
            if r.result_type != "ryuukyoku":
                wins += 1
            h.advance(r)
    return wins, d, sh_sum / max(1, sh_n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--deals", type=int, default=40)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--seed0", type=int, default=555000)
    a = ap.parse_args()

    cks = sorted(glob.glob(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "checkpoints", "dagger_it*.pt")),
        key=os.path.getmtime)
    path = a.ckpt or (cks[-1] if cks else None)
    if not path:
        print("no dagger checkpoint")
        return

    model, ck = load(path)
    enc = O2.ObsEncoder()
    print("=== %s ===" % os.path.basename(path))
    print("  iter=%s" % ck.get("iter"))
    w, d, sh = run(model, enc, a.deals, a.temp, a.seed0)
    print("")
    print("  均匀采样下学生自己的和了率 = %d/%d = %.2f%%"
          % (w, d, 100.0 * w / max(1, d)))
    print("  平均向听 = %.2f" % sh)
    print("")
    print("  参照线:")
    print("    uniform-random  0/200 = 0.00%")
    print("    shanten-greedy 23/200 = 11.50%")
    print("  判据: 学生 > 0% 即证明网络学到了打牌（随机策略永远 0%）")


if __name__ == "__main__":
    main()
