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

from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mahjong import env as E
from mahjong import obs2 as O2
from mahjong import touch as TOUCH
from mahjong.hanchan import HanchanState
from mahjong.net import MahjongNet
from train_hanchan import (encode, slot_of, hand_shanten, N_ACTIONS, SIT_DIM)


def load_model(path):
    """加载半庄线（47 槽）checkpoint。

    2026-11-05 修复：旧版固定按 54 槽 MahjongNet 建网，而默认 glob 选中的
    `a2c_r*_it*.pt` 是 47 槽半庄模型 → `load_state_dict` 直接
    size mismatch（47 vs 54），eval_hanchan.py 开箱即崩。
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ch = ck.get("channels", 64)
    nb = ck.get("blocks", 4)
    model = MahjongNet(channels=ch, n_blocks=nb,
                       n_actions=N_ACTIONS,
                       touch_dim=TOUCH.TOUCH_DIM,
                       extra_scalar=SIT_DIM)
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
    """向听贪心基线：直接复用 `mahjong.policies.GreedyTenpai`。

    2026-11-05 修复：旧版在这里手写了一份「见 RYUUKYOKU 就选」的旧逻辑，
    与已修复的 GreedyTenpai 不一致——一进听就主动宣言流局，和牌率恒 0，
    作为基线上界会低估。现在统一用引擎策略实现，避免两份逻辑漂移。
    """
    from mahjong import policies as POL
    e = hstate.new_env(seed=seed)
    pols = {i: POL.make_policy("greedy_tenpai", seed=seed * 97 + i)
            for i in range(4)}
    r = e.start_turn()
    g = 0
    while r is None and g < 400:
        g += 1
        acts = {}
        for p in e.pending_players():
            legal = e.legal_actions(p)
            acts[p] = (pols[p].choose(e, p, legal) if legal
                       else E.PASS_ACTION)
        r = e.step(acts)
    return r


def run(fn, n_hanchans, label, east_only=False):
    t0 = time.time()
    wins = deals = 0
    yaku_c = Counter()
    for i in range(n_hanchans):
        h = HanchanState(dealer=i % 4, east_only=east_only)
        while not h.finished:
            r = fn(h, 7000 + i * 13 + h.current_index())
            if r is None:
                break
            deals += 1
            if r.result_type != "ryuukyoku":
                wins += 1
                for y in (getattr(r, "yaku", ()) or ()):
                    yaku_c[y] += 1
            h.advance(r)
    dt = time.time() - t0
    pct = 100.0 * wins / max(1, deals)
    tanyao = 100.0 * yaku_c.get("断幺九", 0) / max(1, wins)
    pinfu = 100.0 * yaku_c.get("平和", 0) / max(1, wins)
    seat_pct = pct / 4.0   # 四席同一策略时：局和牌率/4 ≈ 单座位和牌率
    print("%-20s 局和牌=%3d/%3d (%5.2f%%, 座位≈%5.2f%%)  断幺=%4.1f%% "
          "平和=%4.1f%%  %4.0fs"
          % (label, wins, max(1, deals), pct, seat_pct, tanyao, pinfu, dt))
    return pct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--hanchans", type=int, default=3)
    ap.add_argument("--mode", default="hanchan", choices=["hanchan", "east"],
                    help="hanchan=E1..S4 半庄；east=只打东 1..4 东风战")
    a = ap.parse_args()

    # 按 mtime 取最新：文件名 `a2c_r10_*` 字典序排在 `a2c_r9_*` 前面，
    # 旧版 sorted()[-1] 会拿到更老的 r9 权重。
    cks = sorted(glob.glob(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "checkpoints", "a2c_r*_it*.pt")),
        key=os.path.getmtime)
    path = a.ckpt or (cks[-1] if cks else None)

    east = a.mode == "east"
    print(f"=== 模式：{'东风战(E1-E4)' if east else '半庄(E1-S4)'} ===")
    print("=== baselines ===")
    run(play_random, a.hanchans, "uniform-random", east_only=east)
    run(play_shanten_greedy, a.hanchans, "shanten-greedy", east_only=east)

    if path:
        model, ck = load_model(path)
        print("=== model: %s ===" % os.path.basename(path))
        print("  iter=%s deals=%s train_win=%s"
              % (ck.get("iter"), ck.get("deals"), ck.get("win_rate")))
        enc = O2.ObsEncoder()
        run(lambda h, s: play_model(model, enc, h, s),
            a.hanchans, "MODEL (greedy)", east_only=east)
    else:
        print("no checkpoint found")


if __name__ == "__main__":
    main()
