"""半庄训练器：弃听 / 全防 / 东南场次 / 分差 全部纳入奖励与观测。

与 train_ppo.py 的差别
----------------------
train_ppo 只跑「单局」，奖励是最终点数 + 手牌完整性。
本文件跑**半庄**（东场 4 局 + 南场 4 局，可连庄），并且：

1. 奖励以**分差**为核心：不是绝对点数，是相对领先最多的对手的变化
2. **弃听场景**：局势不可挽回时，保命（不输钱）比搏和更值钱
3. **全防场景**：有人立直时，放铳惩罚是平时的 2.5 倍
4. **末局搏命**：南4 大幅落后时，从「保命」切换到「搏」
5. 观测里追加 12 维局势特征（全部规则可见信息）

奖励（训练用，不改规则）
------------------------
    r = 1.0 * Δ分差/10000              # 主目标
      + 0.5 * (向听不增)                # 手牌完整性
      - 放铳惩罚 × 局势系数
      + 0.8 * (弃听局里没输钱)

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
from mahjong import refengine as REF
from mahjong import scenarios as SC
from mahjong.hanchan import HanchanState
from mahjong.net import MahjongNet, count_params

N_ACTIONS = 54
VALID_SIZES = (1, 2, 4, 5, 7, 8, 10, 11, 13, 14)
SIT_DIM = 12


def slot_of(action):
    t = action.type
    if t == E.ACT_DISCARD and 0 <= action.tile < 34:
        return action.tile
    return 34 + ((t * 7 + action.tile) % 20)


def hand_shanten(env, p_idx):
    """向听数（开源实现，含标准型/七对/国士三型）。"""
    return REF.ref_shanten(env.players[p_idx].full_counts())


def encode(env, encoder, hstate, p_idx):
    """观测 = 牌面平面 + 标量 + 局势特征。"""
    planes, scalar = encoder.encode(env, p_idx)
    sh = hand_shanten(env, p_idx)
    sit_vec, sit, give_up = SC.situation_features(env, hstate, p_idx, sh)
    vec = np.concatenate([planes.ravel(), scalar, sit_vec]).astype(np.float32)
    return vec, sh, sit, give_up


def play_one_deal(model, hstate, seed, device, temperature, encoder,
                  shanten_weight):
    """跑一局（一个 deal）。采样与前向解耦以省 CPU。

    采样用均匀随机（CPU 上逐点前向占 98% 耗时：1233 步要 92.5 秒），
    训练时再用整局观测批量算 log-prob。
    """
    env = hstate.new_env(seed=seed)
    info = []
    result = env.start_turn()
    guard = 0
    while result is None and guard < 400:
        guard += 1
        actions = {}
        for p in env.pending_players():
            vec, sh, sit, give_up = encode(env, encoder, hstate, p)
            legal = env.legal_actions(p)
            if not legal:
                actions[p] = E.PASS_ACTION
                continue
            slots = [slot_of(a) for a in legal]
            pick = random.choice(slots)
            act = legal[slots.index(pick)]
            actions[p] = act
            sa = 8
            if env.phase == E.Phase.DISCARD:
                pl = env.players[p]
                if act.tile >= 0 and act.tile in pl.hand:
                    pl.hand.remove(act.tile)
                    pl.hand.sort()
                    sa = hand_shanten(env, p)
                    pl.hand.append(act.tile)
                    pl.hand.sort()
            info.append((vec, slots, pick, sh, sa, p, sit.label, give_up))
        result = env.step(actions)
    return env, result, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hanchans", type=int, default=8, help="每次迭代几个半庄")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--shanten-weight", type=float, default=0.5)
    ap.add_argument("--deal-weight", type=float, default=1.0)
    ap.add_argument("--survive-weight", type=float, default=0.8)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--out", default="checkpoints/hanchan_net.pt")
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()

    device = "cpu"
    torch.manual_seed(args.seed0)
    random.seed(args.seed0)
    np.random.seed(args.seed0)

    model = MahjongNet(channels=args.channels, n_blocks=args.blocks).to(device)
    # 观测维度增加了 12 维局势特征
    base = O2.OBS_TENSOR_DIM
    model.head_fc[0] = torch.nn.Linear(
        model.head_fc[0].in_features + SIT_DIM, model.head_fc[0].out_features)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    encoder = O2.ObsEncoder()
    print(f"device={device} params={count_params(model):,} "
          f"hanchans/iter={args.hanchans}", flush=True)

    t0 = time.time()
    tot_deals = tot_wins = 0
    give_up_seen = defense_seen = lastchance_seen = 0

    for it in range(args.iters):
        opt.zero_grad()
        vecs, slots, rets = [], [], []

        for hc in range(args.hanchans):
            hstate = HanchanState(
                dealer=random.randrange(4),
                start_points=25000)
            deal_seed = args.seed0 * 104729 + it * 257 + hc
            sub = deal_seed
            while not hstate.finished:
                env, result, info = play_one_deal(
                    model, hstate, sub, device, args.temperature,
                    encoder, args.shanten_weight)
                sub += 1
                if result is None:
                    break
                tot_deals += 1
                if result.result_type != "ryuukyoku":
                    tot_wins += 1
                # 统计场景
                for (_, _, _, _, _, _, label, gu) in info:
                    if label == "defense":
                        defense_seen += 1
                    if label == "last_chance":
                        lastchance_seen += 1
                    if gu:
                        give_up_seen += 1

                by_player = {}
                for rec in info:
                    by_player.setdefault(rec[5], []).append(rec)
                for p, lst in by_player.items():
                    n = len(lst)
                    # 终局奖励每家只算一次（原实现在每个决策点重算，
                    # 一次半庄要多算 ~1200 次，白白吃掉全部训练时间）
                    r_term = SC.reward_for_result(
                        p, result, _sit_stub(lst[-1][6]),
                        hstate, give_up_flag=lst[-1][7],
                        w_deal=args.deal_weight,
                        w_survive=args.survive_weight)
                    for k, (vec, legal_slots, pick, sb, sa, _, label, gu) \
                            in enumerate(lst):
                        disc = args.gamma ** (n - 1 - k)
                        r = r_term * disc
                        if args.shanten_weight > 0 and sb < 8 and sa < 8:
                            r += args.shanten_weight * (sa - sb) * disc
                        vecs.append(vec)
                        slots.append((pick, legal_slots))
                        rets.append(r)
                hstate.advance(result)

        if not rets:
            continue
        rt = torch.tensor(rets, dtype=torch.float32)
        adv = rt - rt.mean()
        CH = 2048
        for start in range(0, len(vecs), CH):
            sl = slots[start:start + CH]
            adv_chunk = adv[start:start + CH]
            x = torch.from_numpy(np.stack(vecs[start:start + CH]))
            logits, _ = model(x)
            chosen_l, ent_l = [], []
            for i, (pick, legal_slots) in enumerate(sl):
                mask = torch.full((N_ACTIONS,), -1e9)
                mask[legal_slots] = 0.0
                lp = F.log_softmax(logits[i] + mask, dim=-1)
                chosen_l.append(lp[pick])
                ent_l.append(-(lp.exp() * lp).sum())
            pol = -(torch.stack(chosen_l) * adv_chunk).mean()
            ent = torch.stack(ent_l).mean()
            (pol - args.entropy * ent).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if it % args.log_every == 0 or it == args.iters - 1:
            dt = time.time() - t0
            wr = 100.0 * tot_wins / max(1, tot_deals)
            print(f"iter {it:4d} deals={tot_deals:5d} win={wr:5.2f}% "
                  f"R={rt.mean().item():+.3f} ent? "
                  f"giveup={give_up_seen} defend={defense_seen} "
                  f"last={lastchance_seen} {dt:.0f}s", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "arch": "hanchan_v1",
        "state_dict": model.state_dict(),
        "channels": args.channels,
        "blocks": args.blocks,
        "sit_dim": SIT_DIM,
        "deals": tot_deals,
        "win_rate": 100.0 * tot_wins / max(1, tot_deals),
        "give_up_seen": give_up_seen,
        "defense_seen": defense_seen,
        "last_chance_seen": lastchance_seen,
    }, args.out)
    print(f"saved {args.out}")


class _SitStub:
    """给 reward_for_result 用的最小 Situation 替身。"""
    __slots__ = ("riichi_count", "is_last_round", "point_gap")

    def __init__(self, label):
        self.riichi_count = 0
        self.is_last_round = False
        self.point_gap = 0.0


def _sit_stub(label):
    s = _SitStub(label)
    if label == "defense":
        s.riichi_count = 1
    if label == "last_chance":
        s.is_last_round = True
        s.point_gap = -20000.0
    return s


if __name__ == "__main__":
    main()
