"""评估：把模型和基线放在一起跑，输出对照表。

用途：回答「模型到底学没学到东西」，而不是看训练损失。
判据（可证伪）：
  - 模型和牌率必须显著高于「均匀随机」基线
  - 模型和牌率不应低于「向听贪心」基线太多（贪心用的是向听数，
    属于规则信息，所以它是很强的上界参考）
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mahjong import env as E
from mahjong import obs2 as O2
from mahjong import shanten_ref as SR
from mahjong.net import MahjongNet
from train_ppo import slot_of, N_ACTIONS, hand_shanten, VALID_SIZES


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = MahjongNet().to(device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    print(f"loaded {path} iters={ck.get('iters')} "
          f"reward={ck.get('reward')} "
          f"train_win_rate={ck.get('train_win_rate')}")
    return m, ck


def play_model(model, encoder, seed, device, greedy=True):
    env = E.MahjongEnv(seed=seed, dealer=seed % 4)
    result = env.start_turn()
    g = 0
    while result is None and g < 400:
        g += 1
        pend = {}
        idxs = []
        for p in env.pending_players():
            planes, scalar = encoder.encode(env, p)
            vec = np.concatenate([planes.ravel(), scalar]).astype(np.float32)
            legal = env.legal_actions(p)
            if not legal:
                pend[p] = None
                continue
            pend[p] = (vec, [slot_of(a) for a in legal], legal)
            idxs.append(p)
        logits = {}
        if idxs:
            with torch.no_grad():
                lg, _ = model(torch.from_numpy(
                    np.stack([pend[p][0] for p in idxs])).to(device))
            for k, p in enumerate(idxs):
                logits[p] = lg[k]
        actions = {}
        for p, info in pend.items():
            if info is None:
                actions[p] = E.PASS_ACTION
                continue
            vec, slots, legal = info
            mask = torch.full((N_ACTIONS,), -1e9, device=device)
            mask[slots] = 0.0
            masked = logits[p] + mask
            pick = (int(masked.argmax()) if greedy else
                    int(torch.multinomial(
                        F.softmax(masked, -1), 1)))
            if pick not in slots:
                pick = slots[0]
            actions[p] = legal[slots.index(pick)]
        result = env.step(actions)
    return result


def play_random(seed):
    import random as _r
    rng = _r.Random(seed)
    env = E.MahjongEnv(seed=seed, dealer=seed % 4)
    result = env.start_turn()
    g = 0
    while result is None and g < 400:
        g += 1
        actions = {}
        for p in env.pending_players():
            legal = env.legal_actions(p)
            actions[p] = rng.choice(legal)
        result = env.step(actions)
    return result


def play_shanten_greedy(seed):
    env = E.MahjongEnv(seed=seed, dealer=seed % 4)
    result = env.start_turn()
    g = 0
    while result is None and g < 400:
        g += 1
        actions = {}
        for p in env.pending_players():
            legal = env.legal_actions(p)
            if env.phase != E.Phase.DISCARD:
                actions[p] = next(
                    (a for a in legal if a.type == E.ACT_RON), legal[0])
                continue
            done = False
            for tp in (E.ACT_TSUMO, E.ACT_RON, E.ACT_RYUUKYOKU):
                z = [a for a in legal if a.type == tp]
                if z:
                    actions[p] = z[0]; done = True; break
            if done:
                continue
            disc = [a for a in legal if a.type == E.ACT_DISCARD]
            if not disc:
                actions[p] = legal[0]; continue
            pl = env.players[p]
            best, bv = disc[0], -99
            for a in disc:
                if a.tile in pl.hand:
                    pl.hand.remove(a.tile)
                pl.hand.sort()
                v = -hand_shanten(env, p)
                pl.hand.append(a.tile); pl.hand.sort()
                if v > bv:
                    bv, best = v, a
            actions[p] = best
        result = env.step(actions)
    return result


def evaluate(fn, games, label):
    t0 = time.time()
    w = dr = 0
    pts = [0, 0, 0, 0]
    for i in range(games):
        seed = 5000 + i
        r = fn(seed)
        if r is None:
            continue
        net = E.net_points(r, [25000] * 4)
        for k in range(4):
            pts[k] += net[k]
        if r.result_type == "ryuukyoku":
            dr += 1
        else:
            w += 1
    dt = time.time() - t0
    print(f"{label:22s} win={w:4d}/{games} ({100.0*w/games:5.2f}%) "
          f"draw={dr:4d} avgdiff={[round(x/games) for x in pts]} "
          f"{dt:.0f}s")
    return 100.0 * w / games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/riichi_net_v3.pt")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--skip-baselines", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 76)
    if not args.skip_baselines:
        evaluate(play_random, args.games, "uniform-random")
        evaluate(play_shanten_greedy, args.games, "shanten-greedy")
    model, ck = load_model(args.ckpt, device)
    encoder = O2.ObsEncoder()
    evaluate(lambda s: play_model(model, encoder, s, device,
                                   greedy=not args.sample),
             args.games, "MODEL (greedy)" if not args.sample
             else "MODEL (sample)")
    print("=" * 76)


if __name__ == "__main__":
    main()
