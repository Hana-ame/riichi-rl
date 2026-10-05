from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mahjong import env as E
from mahjong import obs2 as O2
from mahjong import scenarios as SC
from mahjong import touch as TOUCH
from mahjong.net import MahjongNet, count_params
from train_hanchan import (N_ACTIONS, SIT_DIM, encode, hand_shanten,
                            slot_of)


def teacher_pick(env, p_idx, legal, slots):
    """向听贪心教师：每步最大化「打出后的向听数」。

    这是规则层计算（向听数 = 离和牌还差几张牌），不含任何牌效表。
    """
    for t in (E.ACT_TSUMO, E.ACT_RON, E.ACT_RYUUKYOKU):
        for s, a in zip(slots, legal):
            if a.type == t:
                return s, a
    if env.phase == E.Phase.DISCARD:
        disc = [(s, a) for s, a in zip(slots, legal)
                if a.type == E.ACT_DISCARD]
        if disc:
            pl = env.players[p_idx]
            best_s, best_v = disc[0][0], -99
            for s, a in disc:
                if a.tile >= 0 and a.tile in pl.hand:
                    pl.hand.remove(a.tile)
                pl.hand.sort()
                v = -hand_shanten(env, p_idx)
                if a.tile >= 0:
                    pl.hand.append(a.tile)
                pl.hand.sort()
                if v > best_v:
                    best_v, best_s = v, s
            for s, a in zip(slots, legal):
                if s == best_s:
                    return best_s, a
    return random.choice(slots), legal[slots.index(random.choice(slots))]


def student_pick(model, vec, tf, slots, device, eps, temp):
    """学生：eps 概率跟教师，否则按自己的分布采样。"""
    if random.random() < eps:
        return None, None      # 交给教师
    x = torch.from_numpy(vec).unsqueeze(0).to(device)
    tx = torch.from_numpy(tf).unsqueeze(0).to(device)
    with torch.no_grad():
        lg, _ = model(x, tx)
    m = torch.full((N_ACTIONS,), -1e9)
    m[slots] = 0.0
    lg = lg[0] + m
    if temp <= 0:
        pick = int(lg.argmax())
    else:
        pick = int(torch.multinomial(F.softmax(lg / temp, -1), 1))
    if pick not in slots:
        pick = slots[0]
    return pick, None


def collect(model, hstate, encoder, seed, device, eps, temp):
    """跑一个半庄，收集 (观测, 教师动作, 收益) 三元组。"""
    samples = []
    env = hstate.new_env(seed=seed)
    result = env.start_turn()
    guard = 0
    while result is None and guard < 400:
        guard += 1
        actions = {}
        pend = {}
        for p in env.pending_players():
            vec, sh, sit, gu, tf = encode(env, encoder, hstate, p)
            legal = env.legal_actions(p)
            if not legal:
                pend[p] = None
                continue
            slots = [slot_of(a) for a in legal]
            student_slot, _ = student_pick(model, vec, tf, slots, device,
                                         eps, temp)
            if student_slot is None:
                pick, act = teacher_pick(env, p, legal, slots)
            else:
                pick = student_slot
                act = legal[slots.index(pick)]
            pend[p] = (vec, tf, slots, pick, act)
            actions[p] = act
        for p in pend:
            if pend[p] is None:
                actions[p] = E.PASS_ACTION
        result = env.step(actions)
        for p, d in pend.items():
            if d is None:
                continue
            vec, tf, slots, pick, act = d
            samples.append((vec, tf, pick, p))
    # 收益：按最终点数差（归一化）
    if result is None:
        return samples, None, None
    rets = [result.points_delta[i] / 10000.0 for i in range(4)]
    return samples, rets, result


def _new_hstate(it, hc):
    """半庄初始点数：几轮后故意分化，让弃听/全防场景能进训练。"""
    from mahjong.hanchan import HanchanState
    h = HanchanState(dealer=(it + hc) % 4)
    if it >= 3:
        h.sticks = [25000, 25000 - 9000 * (hc % 3), 25000, 25000]
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--hanchans", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eps0", type=float, default=1.0)
    ap.add_argument("--eps-min", type=float, default=0.15)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--bc-weight", type=float, default=1.0)
    ap.add_argument("--adv-weight", type=float, default=0.5)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-deals", type=int, default=30)
    ap.add_argument("--out", default="checkpoints/dagger.pt")
    ap.add_argument("--seed0", type=int, default=0)
    a = ap.parse_args()

    device = "cpu"
    torch.manual_seed(a.seed0)
    random.seed(a.seed0)
    np.random.seed(a.seed0)

    model = MahjongNet(channels=a.channels, n_blocks=a.blocks,
                      n_actions=N_ACTIONS, touch_dim=TOUCH.TOUCH_DIM,
                      extra_scalar=SIT_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    enc = O2.ObsEncoder()
    print("device=%s params=%s" % (device, format(count_params(model), ",")),
          flush=True)

    t0 = time.time()
    for it in range(a.iters):
        frac = it / max(1, a.iters - 1)
        eps = a.eps0 + (a.eps_min - a.eps0) * frac
        opt.zero_grad()
        vecs, tfs, targets, advs = [], [], [], []
        tot_deals = wins = 0
        for hc in range(a.hanchans):
            hstate = _new_hstate(it, hc)
            while not hstate.finished:
                s, rets, result = collect(model, hstate, enc,
                                 a.seed0 * 7919 + it * 257 + hc * 13
                                 + hstate.current_index(), device, eps,
                                 a.temp)
                if rets is None:
                    break
                tot_deals += 1
                for vec, tf, pick, p in s:
                    vecs.append(vec)
                    tfs.append(tf)
                    targets.append(pick)
                    advs.append(rets[p])
                hstate.advance(result)
        # 训练
        if not vecs:
            continue
        x = torch.from_numpy(np.stack(vecs))
        tx = torch.from_numpy(np.stack(tfs))
        tg = torch.tensor(targets, dtype=torch.long)
        rt = torch.tensor(advs, dtype=torch.float32)
        adv = rt - rt.mean()
        logits, values = model(x, tx)
        loss_bc = F.cross_entropy(logits, tg)
        logp = F.log_softmax(logits, -1)
        pol = -(logp.gather(1, tg.unsqueeze(1)).squeeze(1) * adv).mean()
        loss = a.bc_weight * loss_bc + a.adv_weight * pol
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % a.log_every == 0 or it == a.iters - 1:
            print("iter %4d deals=%5d eps=%.2f bc=%.4f pol=%+.4f %4.0fs"
                  % (it, tot_deals, eps, loss_bc.item(), pol.item(),
                     time.time() - t0), flush=True)

        if (it + 1) % a.eval_every == 0 or it == a.iters - 1:
            torch.save({"arch": "dagger_v1",
                        "state_dict": model.state_dict(),
                        "channels": a.channels, "blocks": a.blocks,
                        "touch_dim": TOUCH.TOUCH_DIM,
                        "sit_dim": SIT_DIM, "iter": it + 1},
                       a.out.replace(".pt", "_it%d.pt" % (it + 1)))
    print("done")

if __name__ == "__main__":
    main()
