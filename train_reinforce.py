"""纯 RL 训练器：REINFORCE + 基线 + 熵正则，自对弈。

修复记录：
1. 策略梯度损失改为使用带合法动作 mask 的 `log_softmax`（旧版误用了原始未归一化 `logits`）；
2. 动作槽位从会 20/20 冲突的 54 槽哈希改为无冲突 47 槽显式映射；
3. 优势标准化，并屏蔽听牌时的非法主动流局。
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
from mahjong.net import MahjongNet, count_params
from train_hanchan import (
    N_ACTIONS,
    filter_legal_actions,
    resolve_action_for_slot,
    slot_of,
)


class Step:
    __slots__ = ("vec", "slot", "legal_slots")

    def __init__(self, vec, slot, legal_slots):
        self.vec = vec
        self.slot = slot
        self.legal_slots = legal_slots


def rollout(model, encoder, seed, device, temperature, greedy=False,
            allow_ryuukyoku=False):
    """跑一局，收集所有主体的 (观测, 动作槽, 合法槽位) 与结果。"""
    env = E.MahjongEnv(seed=seed, dealer=seed % 4)
    collected = {i: [] for i in range(4)}
    result = env.start_turn()
    steps = 0
    model.eval()
    while result is None and steps < 400:
        steps += 1
        actions = {}
        for p in env.pending_players():
            planes, scalar = encoder.encode(env, p)
            vec = np.concatenate([planes.ravel(), scalar]).astype(np.float32)
            legal = filter_legal_actions(env.legal_actions(p), allow_ryuukyoku)
            if not legal:
                actions[p] = E.PASS_ACTION
                continue
            slots = [slot_of(a) for a in legal]
            uniq_slots = list(dict.fromkeys(slots))
            with torch.no_grad():
                x = torch.from_numpy(vec).unsqueeze(0).to(device)
                raw_logits = model(x)[0][0]
            mask = torch.full((N_ACTIONS,), -1e9, dtype=torch.float32, device=device)
            mask[uniq_slots] = 0.0
            logits = raw_logits + mask
            if greedy:
                pick_slot = int(logits.argmax().item())
            else:
                probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
                pick_slot = int(torch.multinomial(probs, 1).item())
            if pick_slot not in uniq_slots:
                pick_slot = uniq_slots[0]
            actions[p] = resolve_action_for_slot(legal, slots, pick_slot, raw_logits)
            collected[p].append(Step(vec, pick_slot, uniq_slots))
        result = env.step(actions)
    return result, collected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--per-tile-head", action="store_true")
    ap.add_argument("--allow-ryuukyoku", action="store_true")
    ap.add_argument("--out", default="checkpoints/riichi_net.pt")
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed0)
    random.seed(args.seed0)
    np.random.seed(args.seed0)

    model = MahjongNet(
        channels=args.channels, n_blocks=args.blocks,
        n_actions=N_ACTIONS, per_tile_head=args.per_tile_head,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    encoder = O2.ObsEncoder()
    print(f"device={device}  params={count_params(model):,}")

    t0 = time.perf_counter()
    total_games = wins = draws = 0

    for it in range(args.iters):
        opt.zero_grad()
        all_vecs, all_slots, all_legals, all_rets = [], [], [], []

        for g in range(args.games):
            seed = args.seed0 * 7919 + it * 131 + g
            result, collected = rollout(
                model, encoder, seed, device, args.temperature,
                allow_ryuukyoku=args.allow_ryuukyoku)
            if result is None:
                continue
            total_games += 1
            if result.result_type == "ryuukyoku":
                draws += 1
            else:
                wins += 1
            for i in range(4):
                steps = collected[i]
                if not steps:
                    continue
                adv = result.points_delta[i] / 10000.0
                n = len(steps)
                for k, st in enumerate(steps):
                    ret = adv * (args.gamma ** (n - 1 - k))
                    all_vecs.append(st.vec)
                    all_slots.append(st.slot)
                    all_legals.append(st.legal_slots)
                    all_rets.append(ret)

        if not all_rets:
            continue

        rets = torch.tensor(all_rets, dtype=torch.float32, device=device)
        adv = (rets - rets.mean()) / rets.std().clamp_min(1e-6)

        model.train()
        x = torch.from_numpy(np.stack(all_vecs)).to(device)
        slots = torch.tensor(all_slots, dtype=torch.long, device=device)
        mrow = torch.full((len(all_slots), N_ACTIONS), -1e9, device=device)
        for i, legal_slots in enumerate(all_legals):
            mrow[i, legal_slots] = 0.0

        logits, _ = model(x)
        logp = F.log_softmax(logits + mrow, dim=-1)
        chosen_logp = logp.gather(1, slots.unsqueeze(1)).squeeze(1)
        ent = -(logp.exp() * logp).sum(-1).mean()

        policy_loss = -(chosen_logp * adv).mean()
        loss = policy_loss - args.entropy * ent

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % 20 == 0 or it == args.iters - 1:
            dt = time.perf_counter() - t0
            print(f"iter {it:4d} games={total_games:5d} win={wins:4d} "
                  f"draw={draws:4d} pol={policy_loss.item():+.4f} "
                  f"ent={ent.item():.3f} R̄={rets.mean().item():+.3f} "
                  f"{dt:.0f}s", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "arch": "convformer_v1",
        "state_dict": model.state_dict(),
        "channels": args.channels,
        "blocks": args.blocks,
        "n_actions": N_ACTIONS,
        "n_planes": O2.N_PLANES,
        "scalar_dim": O2.SCALAR_DIM,
        "per_tile_head": bool(args.per_tile_head),
        "iters": args.iters,
    }, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
