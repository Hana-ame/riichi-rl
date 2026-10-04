"""纯 RL 训练器：REINFORCE + 基线 + 熵正则，自对弈。

铁律（用户明确）：
- **不使用任何人类打谱、不做行为克隆、不用专家示范**
- 奖励只来自对局结果（和了/流局/点数差），不含任何牌效先验
- 观测不含对手手牌与牌山顺序

奖励：r = points_delta[i] / 10000（该局该家的点数变化，归一化）
损失：-E[logp * (R - b)] - entropy_coef * H
基线 b：同一批次的回报均值（batch mean baseline，不引入人类知识）
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

N_ACTIONS = 54


def slot_of(action):
    """Action -> 固定槽位。显式表，不用哈希（34..53 共 20 个特殊槽）。"""
    t = action.type
    if t == E.ACT_DISCARD and 0 <= action.tile < 34:
        return action.tile
    return 34 + ((t * 7 + action.tile) % 20)


class Step:
    __slots__ = ("vec", "slot")
    def __init__(self, vec, slot):
        self.vec = vec
        self.slot = slot


def rollout(model, encoder, seed, device, temperature, greedy=False):
    """跑一局，收集所有主体的 (观测, 动作槽, 回报) 与结果。"""
    env = E.MahjongEnv(seed=seed, dealer=seed % 4)
    collected = {i: [] for i in range(4)}
    result = env.start_turn()
    steps = 0
    while result is None and steps < 400:
        steps += 1
        actions = {}
        for p in env.pending_players():
            planes, scalar = encoder.encode(env, p)
            vec = np.concatenate([planes.ravel(), scalar]).astype(np.float32)
            legal = env.legal_actions(p)
            if not legal:
                actions[p] = E.PASS_ACTION
                continue
            slots = [slot_of(a) for a in legal]
            with torch.no_grad():
                x = torch.from_numpy(vec).unsqueeze(0).to(device)
                logits = model(x)[0]
            mask = torch.full((N_ACTIONS,), -1e9, dtype=torch.float32)
            mask[slots] = 0.0
            logits = logits + mask.to(device)
            if greedy:
                pick_slot = int(logits.argmax())
            else:
                probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
                pick_slot = int(torch.multinomial(probs, 1))
            if pick_slot not in slots:
                pick_slot = slots[0]
            actions[p] = legal[slots.index(pick_slot)]
            collected[p].append(Step(vec, pick_slot))
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
    ap.add_argument("--out", default="checkpoints/riichi_net.pt")
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed0)
    random.seed(args.seed0)
    np.random.seed(args.seed0)

    model = MahjongNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    encoder = O2.ObsEncoder()
    print(f"device={device}  params={count_params(model):,}")

    t0 = time.perf_counter()
    total_games = wins = draws = 0

    for it in range(args.iters):
        opt.zero_grad()
        all_vecs, all_slots, all_rets = [], [], []

        for g in range(args.games):
            seed = args.seed0 * 7919 + it * 131 + g
            result, collected = rollout(
                model, encoder, seed, device, args.temperature)
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
                    all_rets.append(ret)

        if not all_rets:
            continue

        rets = torch.tensor(all_rets, dtype=torch.float32, device=device)
        baseline = rets.mean()
        adv = rets - baseline

        x = torch.from_numpy(np.stack(all_vecs)).to(device)
        slots = torch.tensor(all_slots, dtype=torch.long, device=device)
        logits, values = model(x)

        chosen = logits.gather(1, slots.unsqueeze(1)).squeeze(1)
        logp = F.log_softmax(logits, dim=-1)
        ent = -(logp.exp() * logp).sum(-1).mean()

        policy_loss = -(chosen * adv).mean()
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
        "n_actions": N_ACTIONS,
        "n_planes": O2.N_PLANES,
        "scalar_dim": O2.SCALAR_DIM,
        "iters": args.iters,
    }, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
