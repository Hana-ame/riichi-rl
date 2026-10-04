"""半庄模型评估：模型 vs 随机 vs 向听贪心。

判据（可证伪）：
  - 模型和了率必须显著高于均匀随机
  - 若低于向听贪心，说明还没学到东西
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mahjong import env as E
from mahjong import obs2 as O2
from mahjong import touch as TOUCH
from mahjong.hanchan import HanchanState
from mahjong.net import MahjongNet
from train_hanchan import encode, slot_of, hand_shanten

N_ACTIONS = 54


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ch = ck.get("channels", 64)
    nb = ck.get("blocks", 4)
    td = ck.get("touch_dim", TOUCH.TOUCH_DIM)
    need = ck["state_dict"]["head_fc.0.weight"].shape[1]
    probe = MahjongNet(channels=ch, n_blocks=nb, touch_dim=td)
    extra = need - probe.head_fc[0].in_features
    model = MahjongNet(channels=ch, n_blocks=nb, touch_dim=td,
                      extra_scalar=extra)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def play_model(model, enc, hstate, seed):
    e = hstate.new_env(seed=seed)
    r = e.start_turn()
    g = 0
    while r is None and g < 400:
        g += 1
        recs = {}
        for p in e.pending_players():
            vec, sh, sit, gu, tf = encode(e, enc, hstate, p)
            legal = e.legal_actions(p)
            if not legal:
                recs[p] = None
                continue
            recs[p] = (vec, tf, [slot_of(a) for a in legal], legal)
        idxs = [p for p in recs if recs[p] is not None]
        acts = {}
        if idxs:
            x = torch.from_numpy(np.stack([recs[p][0] for p in idxs]))
            tx = torch.from_numpy(np.stack([recs[p][1] for p in idxs]))
            with torch.no_grad():
                lg, _ = model(x, tx)
            for k, p in enumerate(idxs):
                _, _, slots, legal = recs[p]
                m = torch.full((N_ACTIONS,), -1e9)
                m[slots] = 0.0
                acts[p] = legal[slots.index(int((lg[k] + m).argmax()))]
        for p in recs:
            if recs[p] is None:
                acts[p] = E.PASS_ACTION
        r = e.step(acts)
    return r


def play_random(hstate, seed):
    e = hstate.new_env(seed=seed)
    r = e.start_turn()
    g = 0
    while r is None and g < 400:
        g += 1
        acts = {p: random.choice(e.legal_actions(p))
                 for p in e.pending_players()}
        r = e.step(acts)
    return r


def play_shanten_greedy(hstate, seed):
    e = hstate.new_env(seed=seed)
    r = e.start_turn()
    g = 0
    while r is None and g < 400:
        g += 1
        acts = {}
        for p in e.pending_players():
            legal = e.legal_actions(p)
            if e.phase != E.Phase.DISCARD:
                acts[p] = next((a for a in legal
                                 if a.type == E.ACT_RON), legal[0])
                continue
            hit = [a for a in legal
                  if a.type in (E.ACT_TSUMO, E.ACT_RON,
                                E.ACT_RYUUKYOKU)]
            if hit:
                acts[p] = hit[0]
                continue
            disc = [a for a in legal if a.type == E.ACT_DISCARD]
            if not disc:
                acts[p] = legal[0]
                continue
            pl = e.players[p]
            best, bv = disc[0], -99
            for a in disc:
                if a.tile in pl.hand:
                    pl.hand.remove(a.tile)
                pl.hand.sort()
                v = -hand_shanten(e, p)
                pl.hand.append(a.tile)
                pl.hand.sort()
                if v > bv:
                    bv, best = v, a
            acts[p] = best
        r = e.step(acts)
    return r


def run(fn, n_hanchans, label):
    t0 = time.time()
    wins = deals = 0
    for i in range(n_hanchans):
        h = HanchanState(dealer=i % 4)
        while not h.finished:
            r = fn(h, 7000 + i * 13 + h.current_index())
            if r is None:
                break
            deals += 1
            if r.result_type != "ryuukyoku":
                wins += 1
            h.advance(r)
    dt = time.time() - t0
    pct = 100.0 * wins / max(1, deals)
    print("%-20s win=%3d/%3d (%5.2f%%)  %4.0fs"
          % (label, wins, max(1, deals), pct, dt))
    return pct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--hanchans", type=int, default=3)
    a = ap.parse_args()

    cks = sorted(glob.glob(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "checkpoints", "a2c_r*_it*.pt")))
    path = a.ckpt or (cks[-1] if cks else None)

    print("=== baselines ===")
    run(play_random, a.hanchans, "uniform-random")
    run(play_shanten_greedy, a.hanchans, "shanten-greedy")

    if path:
        model, ck = load_model(path)
        print("=== model: %s ===" % os.path.basename(path))
        print("  iter=%s deals=%s train_win=%s"
              % (ck.get("iter"), ck.get("deals"), ck.get("win_rate")))
        enc = O2.ObsEncoder()
        run(lambda h, s: play_model(model, enc, h, s),
            a.hanchans, "MODEL (greedy)")
    else:
        print("no checkpoint found")


if __name__ == "__main__":
    main()
