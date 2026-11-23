"""DAgger 训练器：学生探索状态 + 教师始终打标 + 合法动作掩码。

修复记录：
1. 真正实现 DAgger：执行动作按 eps 混合学生/教师，但训练目标标签**始终为教师动作**
   （旧版在学生决策时存入了学生自己的动作，导致越训越学自己的错误）；
2. `teacher_pick` 移除 `ACT_RYUUKYOKU` 优先级（旧版一听牌就立刻主动流局），
   并修复 fallback 时调用两次 `random.choice(slots)` 导致槽位与动作不匹配的 Bug；
3. 训练时对 `logits` 施加合法动作 mask，避免把概率质量浪费在非法槽位。
"""

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
from mahjong import touch as TOUCH
from mahjong.net import MahjongNet, count_params
from train_hanchan import (
    N_ACTIONS,
    SIT_DIM,
    encode,
    filter_legal_actions,
    hand_shanten,
    resolve_action_for_slot,
    slot_of,
)


def teacher_pick(env, p_idx, legal, slots):
    """向听贪心教师：优先和牌，其次立直，其次最小化打出后的向听数，响应默认 PASS。"""
    for t in (E.ACT_TSUMO, E.ACT_RON):
        for s, a in zip(slots, legal):
            if a.type == t:
                return s, a
    if env.phase == E.Phase.DISCARD:
        for s, a in zip(slots, legal):
            if a.type == E.ACT_RIICHI:
                return s, a
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
            return best_s, resolve_action_for_slot(legal, slots, best_s)
    for s, a in zip(slots, legal):
        if a.type == E.ACT_PASS:
            return s, a
    pick = random.choice(slots)
    return pick, resolve_action_for_slot(legal, slots, pick)


def student_pick(model, vec, tf, slots, device, eps, temp):
    """学生：eps 概率跟教师，否则按自己的分布采样。"""
    if random.random() < eps:
        return None, None      # 交给教师执行
    x = torch.from_numpy(vec).unsqueeze(0).to(device)
    tx = torch.from_numpy(tf).unsqueeze(0).to(device)
    uniq_slots = list(dict.fromkeys(slots))
    with torch.no_grad():
        raw_lg, _ = model(x, tx)
    m = torch.full((N_ACTIONS,), -1e9, device=device)
    m[uniq_slots] = 0.0
    lg = raw_lg[0] + m
    if temp <= 0:
        pick = int(lg.argmax().item())
    else:
        pick = int(torch.multinomial(F.softmax(lg / temp, -1), 1).item())
    if pick not in uniq_slots:
        pick = uniq_slots[0]
    return pick, raw_lg[0]


def collect(model, hstate, encoder, seed, device, eps, temp):
    """跑一局，收集 (观测, touch, 合法槽位, 教师标签, 执行动作, 玩家) 样本。"""
    samples = []
    env = hstate.new_env(seed=seed)
    result = env.start_turn()
    guard = 0
    model.eval()
    while result is None and guard < 400:
        guard += 1
        actions = {}
        pend = {}
        for p in env.pending_players():
            legal = filter_legal_actions(env.legal_actions(p), allow_ryuukyoku=False)
            if not legal:
                pend[p] = None
                continue
            vec, sh, sit, gu, tf = encode(env, encoder, hstate, p)
            slots = [slot_of(a) for a in legal]
            uniq_slots = list(dict.fromkeys(slots))
            # 始终获取教师标签（真 DAgger）
            t_slot, t_act = teacher_pick(env, p, legal, slots)
            student_slot, raw_lg = student_pick(
                model, vec, tf, slots, device, eps, temp)
            if student_slot is None:
                exec_slot, act = t_slot, t_act
            else:
                exec_slot = student_slot
                act = resolve_action_for_slot(legal, slots, exec_slot, raw_lg)
            pend[p] = (vec, tf, uniq_slots, t_slot, exec_slot, act)
            actions[p] = act
        for p in pend:
            if pend[p] is None:
                actions[p] = E.PASS_ACTION
        result = env.step(actions)
        for p, d in pend.items():
            if d is None:
                continue
            vec, tf, uniq_slots, t_slot, exec_slot, _ = d
            samples.append((vec, tf, uniq_slots, t_slot, exec_slot, p))
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
    ap.add_argument("--adv-weight", type=float, default=0.0,
                    help="辅助 RL 权重（纯 DAgger 建议先设 0.0）")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--per-tile-head", action="store_true")
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-deals", type=int, default=30)
    ap.add_argument("--out", default="checkpoints/dagger.pt")
    ap.add_argument("--seed0", type=int, default=0)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed0)
    random.seed(a.seed0)
    np.random.seed(a.seed0)

    model = MahjongNet(
        channels=a.channels, n_blocks=a.blocks,
        n_actions=N_ACTIONS, touch_dim=TOUCH.TOUCH_DIM,
        extra_scalar=SIT_DIM, per_tile_head=a.per_tile_head,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    enc = O2.ObsEncoder()
    print("device=%s params=%s" % (device, format(count_params(model), ",")),
          flush=True)

    t0 = time.time()
    for it in range(a.iters):
        frac = it / max(1, a.iters - 1)
        eps = a.eps0 + (a.eps_min - a.eps0) * frac
        opt.zero_grad()
        vecs, tfs, legals, targets, exec_slots, advs = [], [], [], [], [], []
        tot_deals = 0
        for hc in range(a.hanchans):
            hstate = _new_hstate(it, hc)
            while not hstate.finished:
                s, rets, result = collect(
                    model, hstate, enc,
                    a.seed0 * 7919 + it * 257 + hc * 13 + hstate.current_index(),
                    device, eps, a.temp)
                if rets is None:
                    break
                tot_deals += 1
                for vec, tf, uniq_slots, t_slot, exec_slot, p in s:
                    vecs.append(vec)
                    tfs.append(tf)
                    legals.append(uniq_slots)
                    targets.append(t_slot)
                    exec_slots.append(exec_slot)
                    advs.append(rets[p])
                hstate.advance(result)
        if not vecs:
            continue
        model.train()
        x = torch.from_numpy(np.stack(vecs)).to(device)
        tx = torch.from_numpy(np.stack(tfs)).to(device)
        tg = torch.tensor(targets, dtype=torch.long, device=device)
        ex = torch.tensor(exec_slots, dtype=torch.long, device=device)
        rt = torch.tensor(advs, dtype=torch.float32, device=device)
        adv = (rt - rt.mean()) / rt.std().clamp_min(1e-6)

        mrow = torch.full((len(legals), N_ACTIONS), -1e9, device=device)
        for i, uniq_slots in enumerate(legals):
            mrow[i, uniq_slots] = 0.0

        logits, _ = model(x, tx)
        masked_logits = logits + mrow
        loss_bc = F.cross_entropy(masked_logits, tg)
        logp = F.log_softmax(masked_logits, -1)
        pol = -(logp.gather(1, ex.unsqueeze(1)).squeeze(1) * adv).mean()
        loss = a.bc_weight * loss_bc + a.adv_weight * pol
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % a.log_every == 0 or it == a.iters - 1:
            print("iter %4d deals=%5d eps=%.2f bc=%.4f pol=%+.4f %4.0fs"
                  % (it, tot_deals, eps, loss_bc.item(), pol.item(),
                     time.time() - t0), flush=True)

        if (it + 1) % a.eval_every == 0 or it == a.iters - 1:
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            torch.save({
                "arch": "dagger_v1",
                "state_dict": model.state_dict(),
                "channels": a.channels,
                "blocks": a.blocks,
                "per_tile_head": bool(a.per_tile_head),
                "touch_dim": TOUCH.TOUCH_DIM,
                "sit_dim": SIT_DIM,
                "extra_scalar": SIT_DIM,
                "iter": it + 1,
            }, a.out.replace(".pt", "_it%d.pt" % (it + 1)))
    print("done")


if __name__ == "__main__":
    main()
