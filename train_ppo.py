"""纯 RL 训练器：批量 rollout + 策略梯度 + 熵正则。

修复记录：
1. 替换原 54 槽哈希 slot_of 为无冲突 47 槽显式映射；
2. 修正向听数奖励符号 `(sb - sa)`（旧版 `sa - sb` 写反，在奖励拆牌）；
3. 默认屏蔽听牌时的主动流局动作，防止奖励作弊；
4. 标准化 advantage 并按 chunk 数量归一化梯度。
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
from mahjong import shanten_ref as SR
from mahjong.net import MahjongNet, count_params
from train_hanchan import (
    N_ACTIONS,
    filter_legal_actions,
    resolve_action_for_slot,
    slot_of,
)

VALID_SIZES = (1, 2, 4, 5, 7, 8, 10, 11, 13, 14)


def hand_shanten(env, p_idx):
    c = env.players[p_idx].full_counts()
    if sum(c) not in VALID_SIZES:
        return 8
    try:
        return SR.Shanten.calculate_shanten(c)
    except Exception:
        return 8


class Pending:
    """一个待决策点：观测 + 合法槽位列表。"""
    __slots__ = ("vec", "legal_slots", "shanten_before", "shanten_after",
                 "player")

    def __init__(self, vec, legal_slots, sb, sa, player):
        self.vec = vec
        self.legal_slots = legal_slots
        self.shanten_before = sb
        self.shanten_after = sa
        self.player = player


def rollout(model, encoder, seed, device, temperature, shanten_weight,
            points_scale, allow_ryuukyoku=False):
    """跑一局，记录所有待决策点与结果。"""
    env = E.MahjongEnv(seed=seed, dealer=seed % 4)
    steps_info = []
    result = env.start_turn()
    guard = 0
    while result is None and guard < 400:
        guard += 1
        pend = {}
        for p in env.pending_players():
            sb = hand_shanten(env, p) if env.phase == E.Phase.DISCARD else 8
            planes, scalar = encoder.encode(env, p)
            vec = np.concatenate([planes.ravel(), scalar]).astype(np.float32)
            legal = filter_legal_actions(env.legal_actions(p), allow_ryuukyoku)
            if not legal:
                pend[p] = None
                continue
            slots = [slot_of(a) for a in legal]
            pend[p] = (vec, slots, legal, sb, p)

        idxs = [p for p in pend if pend[p] is not None]
        logits_by_p = {}
        if idxs:
            batch = np.stack([pend[p][0] for p in idxs])
            with torch.no_grad():
                lg, _ = model(torch.from_numpy(batch).to(device))
            for k, p in enumerate(idxs):
                logits_by_p[p] = lg[k]

        actions = {}
        for p, info in pend.items():
            if info is None:
                actions[p] = E.PASS_ACTION
                continue
            vec, slots, legal, sb, _ = info
            uniq_slots = list(dict.fromkeys(slots))
            mask = torch.full((N_ACTIONS,), -1e9, device=device)
            mask[uniq_slots] = 0.0
            logits = logits_by_p[p] + mask
            logp_all = F.log_softmax(logits / max(temperature, 1e-6), dim=-1)
            probs = logp_all.exp()
            pick = int(torch.multinomial(probs, 1).item())
            if pick not in uniq_slots:
                pick = uniq_slots[0]
            old_lp = float(logp_all[pick].item())
            act = resolve_action_for_slot(legal, slots, pick, logits_by_p[p])
            actions[p] = act

            sa = 8
            if env.phase == E.Phase.DISCARD:
                pl = env.players[p]
                tile = act.tile
                if tile >= 0 and tile in pl.hand:
                    pl.hand.remove(tile)
                    pl.hand.sort()
                    sa = hand_shanten(env, p)
                    pl.hand.append(tile)
                    pl.hand.sort()
            steps_info.append((vec, uniq_slots, pick, sb, sa, p, old_lp))

        result = env.step(actions)
    return env, result, steps_info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--shanten-weight", type=float, default=0.0)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--ppo-epochs", type=int, default=2)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--points-scale", type=float, default=10000.0)
    ap.add_argument("--out", default="checkpoints/riichi_net.pt")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--per-tile-head", action="store_true")
    ap.add_argument("--allow-ryuukyoku", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.device == "cpu":
        device = "cpu"
    elif args.device == "cuda":
        device = "cuda"
    else:
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
    print(f"device={device} params={count_params(model):,} "
          f"games/iter={args.games} shanten_w={args.shanten_weight}",
          flush=True)

    t0 = time.perf_counter()
    total = wins = draws = 0

    for it in range(args.iters):
        opt.zero_grad()
        vecs, slots, old_lps, step_rewards, dones = [], [], [], [], []
        model.eval()

        for g in range(args.games):
            seed = args.seed0 * 7919 + it * 131 + g
            env_g, result, info = rollout(
                model, encoder, seed, device,
                args.temperature, args.shanten_weight,
                args.points_scale, allow_ryuukyoku=args.allow_ryuukyoku)
            if result is None:
                continue
            total += 1
            if result.result_type == "ryuukyoku":
                draws += 1
            else:
                wins += 1

            net_deltas = E.net_points(result, env_g.initial_points)
            by_player = {}
            for vec, legal_slots, pick, sb, sa, p, old_lp in info:
                by_player.setdefault(p, []).append((vec, legal_slots, pick, sb, sa, old_lp))
            for p, lst in by_player.items():
                terminal = net_deltas[p] / args.points_scale
                n = len(lst)
                for k, (vec, legal_slots, pick, sb, sa, old_lp) in enumerate(lst):
                    sr = 0.0
                    if args.shanten_weight > 0 and sb < 8 and sa < 8:
                        sr += args.shanten_weight * (sb - sa)
                    if k == n - 1:
                        sr += terminal
                    vecs.append(vec)
                    slots.append((pick, legal_slots))
                    old_lps.append(old_lp)
                    step_rewards.append(sr)
                    dones.append(k == n - 1)

        if not vecs:
            continue

        CH = 256
        with torch.no_grad():
            vals_list = []
            for start in range(0, len(vecs), CH):
                x = torch.from_numpy(np.stack(vecs[start:start + CH])).to(device)
                _, v = model(x)
                vals_list.append(v.cpu().numpy())
            vals_np = np.concatenate(vals_list, axis=0)

        n_total = len(step_rewards)
        gae_adv = np.zeros(n_total, dtype=np.float32)
        gae_ret = np.zeros(n_total, dtype=np.float32)
        gae_run = 0.0
        for idx_t in range(n_total - 1, -1, -1):
            next_v = 0.0 if dones[idx_t] or idx_t + 1 >= n_total else float(vals_np[idx_t + 1])
            mask_nt = 0.0 if dones[idx_t] else 1.0
            delta_t = float(step_rewards[idx_t]) + args.gamma * next_v * mask_nt - float(vals_np[idx_t])
            gae_run = delta_t + args.gamma * args.gae_lambda * mask_nt * gae_run
            gae_adv[idx_t] = gae_run
            gae_ret[idx_t] = gae_run + float(vals_np[idx_t])

        rt = torch.from_numpy(gae_ret).to(device)
        adv = torch.from_numpy(gae_adv).to(device)
        adv = (adv - adv.mean()) / adv.std().clamp_min(1e-6)
        old_lp_t = torch.tensor(old_lps, dtype=torch.float32, device=device)

        model.train()
        pol_sum = ent_sum = 0.0
        n_chunks = max(1, (len(vecs) + CH - 1) // CH)
        for _ep in range(max(1, args.ppo_epochs)):
            opt.zero_grad()
            pol_ep = ent_ep = 0.0
            for start in range(0, len(vecs), CH):
                sl = slots[start:start + CH]
                adv_chunk = adv[start:start + CH]
                rt_chunk = rt[start:start + CH]
                old_lp_chunk = old_lp_t[start:start + CH]
                x = torch.from_numpy(np.stack(vecs[start:start + CH])).to(device)
                logits, values = model(x)
                mrow = torch.full((len(sl), N_ACTIONS), -1e9, device=device)
                for i, (_, legal_slots) in enumerate(sl):
                    mrow[i, legal_slots] = 0.0
                pick_t = torch.tensor([p for (p, _) in sl], dtype=torch.long, device=device)
                logp = F.log_softmax(logits + mrow, dim=-1)
                chosen_t = logp.gather(1, pick_t.unsqueeze(1)).squeeze(1)
                ent_t = -(logp.exp() * logp).sum(-1).mean()
                ratio = torch.exp((chosen_t - old_lp_chunk).clamp(-10.0, 10.0))
                clipped = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip)
                chunk_pol = -torch.min(ratio * adv_chunk, clipped * adv_chunk).mean()
                chunk_v = F.mse_loss(values, rt_chunk)
                ((chunk_pol + 0.5 * chunk_v - args.entropy * ent_t) / n_chunks).backward()
                pol_ep += chunk_pol.item() / n_chunks
                ent_ep += ent_t.item() / n_chunks
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            pol_sum, ent_sum = pol_ep, ent_ep
        pol_val, ent_val = pol_sum, ent_sum

        if it % args.log_every == 0 or it == args.iters - 1:
            dt = time.perf_counter() - t0
            rate = 100.0 * wins / max(1, total)
            print(f"iter {it:5d} games={total:6d} win={wins:5d} "
                  f"({rate:5.2f}%) draw={draws:6d} pol={pol_val:+.4f} "
                  f"ent={ent_val:.3f} R={rt.mean().item():+.3f} "
                  f"{dt:.0f}s ({total/dt:.1f} g/s)", flush=True)

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
        "reward": (f"points({args.points_scale}) + "
                   f"shanten({args.shanten_weight})"),
        "train_games": total,
        "train_win_rate": 100.0 * wins / max(1, total),
    }, args.out)
    print(f"saved {args.out}  train_win_rate="
          f"{100.0*wins/max(1,total):.2f}%")


if __name__ == "__main__":
    main()
