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
            probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
            pick = int(torch.multinomial(probs, 1).item())
            if pick not in uniq_slots:
                pick = uniq_slots[0]
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
            steps_info.append((vec, uniq_slots, pick, sb, sa, p))

        result = env.step(actions)
    return result, steps_info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--shanten-weight", type=float, default=0.3)
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
        vecs, slots, rets = [], [], []
        model.eval()

        for g in range(args.games):
            seed = args.seed0 * 7919 + it * 131 + g
            result, info = rollout(
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

            by_player = {}
            for vec, legal_slots, pick, sb, sa, p in info:
                by_player.setdefault(p, []).append((vec, legal_slots,
                                                    pick, sb, sa))
            for p, lst in by_player.items():
                terminal = result.points_delta[p] / args.points_scale
                n = len(lst)
                running_ret = terminal
                step_rets = [0.0] * n
                for k in range(n - 1, -1, -1):
                    _, _, _, sb, sa = lst[k]
                    sh_r = 0.0
                    if args.shanten_weight > 0 and sb < 8 and sa < 8:
                        sh_r = args.shanten_weight * (sb - sa)
                    running_ret = sh_r + (args.gamma * running_ret if k < n - 1 else terminal)
                    step_rets[k] = running_ret
                for k, (vec, legal_slots, pick, sb, sa) in enumerate(lst):
                    vecs.append(vec)
                    slots.append((pick, legal_slots))
                    rets.append(step_rets[k])

        if not rets:
            continue
        rt = torch.tensor(rets, dtype=torch.float32, device=device)
        adv = (rt - rt.mean()) / rt.std().clamp_min(1e-6)

        model.train()
        opt.zero_grad()
        pol_sum = 0.0
        ent_sum = 0.0
        CH = 256
        n_chunks = max(1, (len(vecs) + CH - 1) // CH)
        for start in range(0, len(vecs), CH):
            sl = slots[start:start + CH]
            adv_chunk = adv[start:start + CH]
            x = torch.from_numpy(np.stack(vecs[start:start + CH])).to(device)
            logits, _ = model(x)
            mrow = torch.full((len(sl), N_ACTIONS), -1e9, device=device)
            for i, (_, legal_slots) in enumerate(sl):
                mrow[i, legal_slots] = 0.0
            pick_t = torch.tensor([p for (p, _) in sl], dtype=torch.long, device=device)
            logp = F.log_softmax(logits + mrow, dim=-1)
            chosen_t = logp.gather(1, pick_t.unsqueeze(1)).squeeze(1)
            ent_t = -(logp.exp() * logp).sum(-1).mean()
            chunk_pol = -(chosen_t * adv_chunk).mean()
            ((chunk_pol - args.entropy * ent_t) / n_chunks).backward()
            pol_sum += chunk_pol.item() / n_chunks
            ent_sum += ent_t.item() / n_chunks
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
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
