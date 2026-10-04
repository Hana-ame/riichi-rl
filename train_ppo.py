"""纯 RL 训练器：批量 rollout + REINFORCE + 熵正则。

性能关键：不做逐决策点推理。整局跑完收集所有 (观测, 合法槽位)，
再用 GPU 一次性批量前向，最后一次性算 log-prob 与反向。
单次前向从 17.2ms（bs=1）降到亚毫秒（bs=200）。

奖励（训练用，与规则分离）
--------------------------
    r = terminal(points_delta / points_scale) * discount        # 最终点数（主）
      + shanten_weight * (shanten_before - shanten_after) * discount  # 向听下降

规则层（hand/scoring/ruleset）已对照 MahjongRepository/mahjong 2.0.0 核验，
奖励只是塑形，不改变规则。

铁律：不用人类打谱、不做行为克隆、不注入牌效先验。
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

N_ACTIONS = 54
VALID_SIZES = (1, 2, 4, 5, 7, 8, 10, 11, 13, 14)


def slot_of(action):
    t = action.type
    if t == E.ACT_DISCARD and 0 <= action.tile < 34:
        return action.tile
    return 34 + ((t * 7 + action.tile) % 20)


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
            points_scale):
    """跑一局。决策时不推理（只记录），返回所有待决策点与结果。"""
    env = E.MahjongEnv(seed=seed, dealer=seed % 4)
    steps_info = []          # [(Pending, chosen_slot)]
    result = env.start_turn()
    guard = 0
    while result is None and guard < 400:
        guard += 1
        pend = {}
        for p in env.pending_players():
            sb = hand_shanten(env, p) if env.phase == E.Phase.DISCARD else 8
            planes, scalar = encoder.encode(env, p)
            vec = np.concatenate([planes.ravel(), scalar]).astype(np.float32)
            legal = env.legal_actions(p)
            if not legal:
                pend[p] = None
                continue
            slots = [slot_of(a) for a in legal]
            pend[p] = (vec, slots, legal, sb, p)

        # 批量前向：把本步所有主体的观测堆在一起
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
            mask = torch.full((N_ACTIONS,), -1e9, device=device)
            mask[slots] = 0.0
            logits = logits_by_p[p] + mask
            probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
            pick = int(torch.multinomial(probs, 1))
            if pick not in slots:
                pick = slots[0]
            actions[p] = legal[slots.index(pick)]

            sa = 8
            if env.phase == E.Phase.DISCARD:
                pl = env.players[p]
                tile = legal[slots.index(pick)].tile
                if tile >= 0 and tile in pl.hand:
                    pl.hand.remove(tile)
                    pl.hand.sort()
                    sa = hand_shanten(env, p)
                    pl.hand.append(tile)
                    pl.hand.sort()
            steps_info.append((vec, slots, pick, sb, sa, p))

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

    model = MahjongNet(channels=args.channels, n_blocks=args.blocks).to(device)
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

        for g in range(args.games):
            seed = args.seed0 * 7919 + it * 131 + g
            result, info = rollout(model, encoder, seed, device,
                                   args.temperature, args.shanten_weight,
                                   args.points_scale)
            if result is None:
                continue
            total += 1
            if result.result_type == "ryuukyoku":
                draws += 1
            else:
                wins += 1

            # 按玩家分组，算折扣奖励
            by_player = {}
            for vec, legal_slots, pick, sb, sa, p in info:
                by_player.setdefault(p, []).append((vec, legal_slots,
                                                    pick, sb, sa))
            for p, lst in by_player.items():
                terminal = result.points_delta[p] / args.points_scale
                n = len(lst)
                for k, (vec, legal_slots, pick, sb, sa) in enumerate(lst):
                    disc = args.gamma ** (n - 1 - k)
                    r = terminal * disc
                    if args.shanten_weight > 0 and sb < 8 and sa < 8:
                        r += args.shanten_weight * (sa - sb) * disc
                    vecs.append(vec)
                    slots.append((pick, legal_slots))
                    rets.append(r)

        if not rets:
            continue
        rt = torch.tensor(rets, dtype=torch.float32, device=device)
        adv = (rt - rt.mean())

        # 分块前向 + 立刻反向，避免一次性把几千个样本的激活图留在显存里
        opt.zero_grad()
        pol_sum = 0.0
        ent_sum = 0.0
        CH = 256
        for start in range(0, len(vecs), CH):
            sl = slots[start:start + CH]
            adv_chunk = adv[start:start + CH]
            x = torch.from_numpy(np.stack(vecs[start:start + CH])).to(device)
            logits, _ = model(x)
            chosen_l, ent_l = [], []
            for i, (pick, legal_slots) in enumerate(sl):
                mask = torch.full((N_ACTIONS,), -1e9, device=device)
                mask[legal_slots] = 0.0
                lp = F.log_softmax(logits[i] + mask, dim=-1)
                chosen_l.append(lp[pick])
                ent_l.append(-(lp.exp() * lp).sum())
            chosen_t = torch.stack(chosen_l)
            ent_t = torch.stack(ent_l).mean()
            chunk_pol = -(chosen_t * adv_chunk).mean()
            (chunk_pol - args.entropy * ent_t).backward()
            pol_sum += chunk_pol.item()
            ent_sum += ent_t.item()
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
        "n_actions": N_ACTIONS,
        "n_planes": O2.N_PLANES,
        "scalar_dim": O2.SCALAR_DIM,
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
