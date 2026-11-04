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
from mahjong import aux_tasks as AUX
from mahjong import scenarios as SC
from mahjong import touch as TOUCH
from mahjong.hanchan import HanchanState
from mahjong.net import MahjongNet, count_params

N_ACTIONS = 47   # 34 打牌 + 13 特殊（含吃的 3 槽）
VALID_SIZES = (1, 2, 4, 5, 7, 8, 10, 11, 13, 14)
SIT_DIM = 12


#: 特殊动作的固定槽位（2026-10-04 重写）
SLOT_TSUMO = 34
SLOT_RON = 35
SLOT_RIICHI = 36
SLOT_RYUUKYOKU = 37
SLOT_NAGASHI = 38
SLOT_PON = 39
SLOT_MINKAN = 40
SLOT_ADD_KAN = 41
SLOT_ANKAN = 42
SLOT_PASS = 43
SLOT_CHI_BASE = 44          # 吃占 44..46（三种起手）

SLOTS = {
    E.ACT_TSUMO: SLOT_TSUMO,
    E.ACT_RON: SLOT_RON,
    E.ACT_RIICHI: SLOT_RIICHI,
    E.ACT_RYUUKYOKU: SLOT_RYUUKYOKU,
    E.ACT_NAGASHI: SLOT_NAGASHI,
    E.ACT_PON: SLOT_PON,
    E.ACT_MINKAN: SLOT_MINKAN,
    E.ACT_ADD_KAN: SLOT_ADD_KAN,
    E.ACT_ANKAN: SLOT_ANKAN,
    E.ACT_PASS: SLOT_PASS,
}

SLOT_HINT = (
    "此前的 slot_of 用 34 + (type*7+tile) % 20 哈希，把 11 种动作类型",
    "折叠进 20 个槽——实测 20 个槽**全部冲突**。实测策略 96% 的动作",
    "落在 slot 50（PASS 的槽），因为 PASS 在响应阶段总是合法、",
    "是唯一零成本动作，早期策略必然倾向选它 -> 永不推进 -> 和了率 0%。",
    "改为显式分配：每种动作类型一个固定槽，吃占 3 个（三种起手）。",
)


def slot_of(action):
    t = action.type
    if t == E.ACT_DISCARD and 0 <= action.tile < 34:
        return action.tile
    if t == E.ACT_CHI:
        return SLOT_CHI_BASE + (action.tile % 3)
    # 立直 / 暗杠 / 加杠 / 明杠 都是「带 tile 参数」的动作，
    # 但同帧内只可能出现其中一个（响应阶段互斥），
    # 所以可以共用固定槽；只有同帧可能出现多个变体的才需要分开。
    return SLOTS.get(t, SLOT_PASS)


def hand_shanten(env, p_idx):
    """向听数（开源实现，含标准型/七对/国士三型）。"""
    return REF.ref_shanten(env.players[p_idx].full_counts())


def encode(env, encoder, hstate, p_idx):
    """观测 = 牌面平面 + 标量 + 局势特征；另给手牌形状特征（touch）。"""
    planes, scalar = encoder.encode(env, p_idx)
    sh = hand_shanten(env, p_idx)
    sit_vec, sit, give_up = SC.situation_features(env, hstate, p_idx, sh)
    vec = np.concatenate([planes.ravel(), scalar, sit_vec]).astype(np.float32)
    c = env.players[p_idx].full_counts()
    melds = len(env.players[p_idx].melds)
    waits = 0
    if sh == 0:
        for t in range(34):
            if REF.can_win_with(c, t, melds):
                waits += 1
    tf = TOUCH.shape_features(c, melds, sh, waits)
    return vec, sh, sit, give_up, tf


# 对照实验开关（2026-10-04）
FORCE_WIN = False
BEHAVIOUR = "random"


def _pick_behaviour(env_, p, legal, slots):
    """按 BEHAVIOUR 选动作，返回 (slot, action)。"""
    if FORCE_WIN:
        ws = [s for s, a in zip(slots, legal)
              if a.type in (E.ACT_TSUMO, E.ACT_RON)]
        if ws:
            return ws[0], legal[slots.index(ws[0])]
    if BEHAVIOUR == "shanten" and env_.phase == E.Phase.DISCARD:
        ts = [a for a in legal
              if a.type in (E.ACT_TSUMO, E.ACT_RYUUKYOKU)]
        if ts:
            return slot_of(ts[0]), ts[0]
        disc = [(s, a) for s, a in zip(slots, legal)
                if a.type == E.ACT_DISCARD]
        if disc:
            pl = env_.players[p]
            best, bv = disc[0][0], -99
            for s, a in disc:
                if a.tile in pl.hand:
                    pl.hand.remove(a.tile)
                pl.hand.sort()
                v = -hand_shanten(env_, p)
                pl.hand.append(a.tile)
                pl.hand.sort()
                if v > bv:
                    bv, best = v, s
            return best, legal[slots.index(best)]
    pick = random.choice(slots)
    return pick, legal[slots.index(pick)]


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
            vec, sh, sit, give_up, tf = encode(env, encoder, hstate, p)
            legal = env.legal_actions(p)
            if not legal:
                actions[p] = E.PASS_ACTION
                continue
            slots = [slot_of(a) for a in legal]
            pick, act = _pick_behaviour(env, p, legal, slots)
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
            info.append((vec, slots, pick, sh, sa, p, sit.label,
                          give_up, tf))
        result = env.step(actions)
    return env, result, info


def _parse_aux_weights(text):
    """'tanyao=2,pinfu=2' -> {"tanyao": 2.0, "pinfu": 2.0}。

    同一个模型用这几个权重调「更爱断幺还是更爱平和」——不再拆成不同模型/人。
    """
    out = {}
    for part in (text or "").split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        try:
            out[k.strip()] = float(v)
        except ValueError:
            pass
    return out


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
    ap.add_argument("--aux-bonus-weight", type=float, default=0.0,
                    help="额外任务奖励权重（>0 时把每局任务完成奖励加进 reward）")
    ap.add_argument("--aux-weights", default="",
                    help="风格任务权重，如 'tanyao=2,pinfu=2'（同一模型调参）")
    ap.add_argument("--init", default="",
                    help="从已有 ckpt 继续训练（同一模型微调风格权重用）")
    ap.add_argument("--out", default="checkpoints/hanchan_net.pt")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--force-win", action="store_true")
    ap.add_argument("--behaviour", default="random",
                    choices=["random", "shanten"])
    args = ap.parse_args()

    global FORCE_WIN, BEHAVIOUR
    FORCE_WIN = args.force_win
    BEHAVIOUR = args.behaviour
    device = "cpu"
    torch.manual_seed(args.seed0)
    random.seed(args.seed0)
    np.random.seed(args.seed0)

    model = MahjongNet(channels=args.channels, n_blocks=args.blocks,
                      n_actions=N_ACTIONS,
                      touch_dim=TOUCH.TOUCH_DIM).to(device)
    # 观测维度增加了 12 维局势特征
    base = O2.OBS_TENSOR_DIM
    model.head_fc[0] = torch.nn.Linear(
        model.head_fc[0].in_features + SIT_DIM, model.head_fc[0].out_features)
    if args.init:
        _ck = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(_ck["state_dict"])
        print(f"从 {args.init} 继续训练"
              f"（iter={_ck.get('iter')}, win={_ck.get('win_rate')}）",
              flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    encoder = O2.ObsEncoder()
    print(f"device={device} params={count_params(model):,} "
          f"hanchans/iter={args.hanchans}", flush=True)

    t0 = time.time()
    tot_deals = tot_wins = 0
    give_up_seen = defense_seen = lastchance_seen = 0

    for it in range(args.iters):
        opt.zero_grad()
        vecs, slots, rets, touches = [], [], [], []

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
                for rec in info:
                    label, gu = rec[6], rec[7]
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
                    # 额外任务奖励（用户要求：每个模型都给额外任务，完成也有奖励）
                    if args.aux_bonus_weight > 0:
                        _w = _parse_aux_weights(args.aux_weights)
                        _, _aux = AUX.deal_bonus(env, result, p, weights=_w)
                        r_term += args.aux_bonus_weight * _aux / 1000.0
                    for k, (vec, legal_slots, pick, sb, sa, _, label, gu, tf) \
                            in enumerate(lst):
                        disc = args.gamma ** (n - 1 - k)
                        r = r_term * disc
                        if args.shanten_weight > 0 and sb < 8 and sa < 8:
                            r += args.shanten_weight * (sa - sb) * disc
                        vecs.append(vec)
                        touches.append(tf)
                        slots.append((pick, legal_slots))
                        rets.append(r)
                hstate.advance(result)

        if not rets:
            continue
        rt = torch.tensor(rets, dtype=torch.float32)
        # 优势 = 回报 - **价值基线**（不是 batch 均值）。
        # 纯 REINFORCE 用 batch 均值做基线时，一个半庄里所有决策点
        # 共享同一个终局回报，优势几乎全为 0 -> 学不动。
        # 用 critic 估计每个状态的 V(s) 才能把稀疏奖励摊到每一步。
        CH = 2048
        _pol_acc = _v_acc = _ent_acc = 0.0
        _nb = 0
        for start in range(0, len(vecs), CH):
            sl = slots[start:start + CH]
            x = torch.from_numpy(np.stack(vecs[start:start + CH]))
            tx = torch.from_numpy(np.stack(touches[start:start + CH]))
            rt_chunk = rt[start:start + CH]
            # 向量化掩码：不再逐样本建 (N_ACTIONS,) 的 mask（那是主要开销）
            bsz = len(sl)
            mrow = torch.full((bsz, N_ACTIONS), -1e9)
            for i, (pick, legal_slots) in enumerate(sl):
                for s in legal_slots:
                    mrow[i, s] = 0.0
            pick_t = torch.tensor([p for (p, _) in sl], dtype=torch.long)
            logits, values = model(x, tx)
            masked = logits + mrow
            logp_all = F.log_softmax(masked, dim=-1)
            logp_pick = logp_all.gather(1, pick_t.unsqueeze(1)).squeeze(1)
            ent = -(logp_all.exp() * logp_all).sum(-1).mean()
            adv_chunk = rt_chunk - values.detach()
            pol = -(logp_pick * adv_chunk).mean()
            vloss = F.mse_loss(values, rt_chunk)
            (pol + vloss - args.entropy * ent).backward()
            _pol_acc += pol.item()
            _v_acc += vloss.item()
            _ent_acc += ent.item()
            _nb += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # 每轮存一次 checkpoint，避免长跑时一颗都产不出
        if args.save_every and (it + 1) % args.save_every == 0:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save({
                "arch": "hanchan_a2c",
                "state_dict": model.state_dict(),
                "channels": args.channels,
                "blocks": args.blocks,
                "touch_dim": TOUCH.TOUCH_DIM,
                "sit_dim": SIT_DIM,
        "extra_scalar": SIT_DIM,
                "extra_scalar": SIT_DIM,
                "iter": it + 1,
                "deals": tot_deals,
                "win_rate": 100.0 * tot_wins / max(1, tot_deals),
            }, args.out.replace(".pt", "_it%d.pt" % (it + 1)))
        if it % args.log_every == 0 or it == args.iters - 1:
            dt = time.time() - t0
            wr = 100.0 * tot_wins / max(1, tot_deals)
            nb = max(1, _nb)
            print(f"iter {it:4d} deals={tot_deals:5d} win={wr:5.2f}% "
                  f"R={rt.mean().item():+.3f} "
                  f"pol={_pol_acc / nb:+.4f} vloss={_v_acc / nb:.4f} "
                  f"ent={_ent_acc / nb:.3f} "
                  f"giveup={give_up_seen} defend={defense_seen} "
                  f"last={lastchance_seen} {dt:.0f}s", flush=True)
            _pol_acc = _v_acc = _ent_acc = 0.0
            _nb = 0

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "arch": "hanchan_v1",
        "state_dict": model.state_dict(),
        "channels": args.channels,
        "blocks": args.blocks,
        "sit_dim": SIT_DIM,
        "extra_scalar": SIT_DIM,
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
