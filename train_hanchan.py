"""半庄训练器：真正 on-policy 批量采样 PPO + Critic Warmup + BC KL 锚定。

修复记录（解决「从 BC 起步 RL 单调退化 / 纯 RL 训不出来」）：
1. 采样改用当前策略网络 `model`（多半庄并行拼 batch 前向推理），
   彻底修复旧版 `play_one_deal` 用纯均匀随机采样却做 on-policy PPO 的致命 Bug；
2. 修正向听塑形符号 `(sb - sa)`（旧版 `sa - sb` 写反，在奖励 AI 拆牌）；
3. 默认屏蔽听牌时的非法主动流局（ACT_RYUUKYOKU / ACT_NAGASHI），防止奖励作弊；
4. 多张可立直切牌时按对应切牌槽 logit 择优，不再硬选编号最小牌；
5. PPO 使用真实 rollout 时的 `old_logp`、标准化 advantage、梯度按 chunk 数归一；
6. 从 `--init` BC 权重起步时，默认开启 `--critic-warmup`（先冻结策略头训 Critic）
   和 `--kl-coef`（到初始 BC 策略的 KL 锚定），避免初期价值噪声把 BC 策略拉崩。
"""

from __future__ import annotations

import argparse
import copy
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
from mahjong import fast_cache as FC
from mahjong.mp_env import ParallelDealPool
from mahjong import aux_tasks as AUX
from mahjong import scenarios as SC
from mahjong import touch as TOUCH
from mahjong.hanchan import HanchanState, placement_bonus
from mahjong.net import MahjongNet, count_params
from mahjong.dual_net import DualStreamMahjongNet
from mahjong import event_stream as ES

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
    if t == E.ACT_DISCARD and action.tile is not None and 0 <= action.tile < 34:
        return action.tile
    if t == E.ACT_CHI:
        # 按被吃牌在顺子中的相对位置映射：0=左吃(吃最小张), 1=中吃(嵌张), 2=右吃(吃最大张)
        if action.arg is not None and action.arg >= 0 and action.tile is not None:
            rel = max(0, min(2, int(action.arg) - int(action.tile)))
        else:
            rel = int(action.tile) % 3 if action.tile is not None else 0
        return SLOT_CHI_BASE + rel
    return SLOTS.get(t, SLOT_PASS)


def filter_legal_actions(legal, allow_ryuukyoku=False):
    """默认屏蔽听牌时可随时触发的非法主动流局，避免 RL 刷流局分。"""
    if allow_ryuukyoku or not legal:
        return legal
    filtered = [a for a in legal
                if a.type not in (E.ACT_RYUUKYOKU, E.ACT_NAGASHI)]
    return filtered if filtered else legal


def resolve_action_for_slot(legal, slots, pick_slot, logits_row=None):
    """把选中的槽位还原为具体 Action；若同槽有多个立直切牌候选，按切牌 logit 择优。"""
    cands = [a for s, a in zip(slots, legal) if s == pick_slot]
    if not cands:
        return legal[0]
    if len(cands) == 1 or logits_row is None:
        return cands[0]
    if pick_slot == SLOT_RIICHI:
        best_a = max(
            cands,
            key=lambda a: float(logits_row[a.tile]) if 0 <= a.tile < 34 else -1e9,
        )
        return best_a
    return cands[0]


def hand_shanten(env, p_idx):
    """带 LRU 缓存的向听数计算（含标准型/七对/国士三型）。"""
    return FC.cached_shanten(tuple(env.players[p_idx].full_counts()))


def encode(env, encoder, hstate, p_idx):
    """观测 = 牌面平面 + 标量 + 局势特征；另给手牌形状特征（touch）。"""
    planes, scalar = encoder.encode(env, p_idx)
    sh = hand_shanten(env, p_idx)
    sit_vec, sit, give_up = SC.situation_features(env, hstate, p_idx, sh)
    vec = np.concatenate([planes.ravel(), scalar, sit_vec]).astype(np.float32)
    c = env.players[p_idx].full_counts()
    melds = len(env.players[p_idx].melds)
    waits = FC.fast_wait_count(tuple(c)) if sh == 0 else 0
    tf = TOUCH.shape_features(c, melds, sh, waits)
    return vec, sh, sit, give_up, tf


# 行为策略开关：默认 "policy"（真正使用当前模型 on-policy 采样）
FORCE_WIN = False
BEHAVIOUR = "policy"


def _pick_behaviour(env_, p, legal, slots, logits_row=None, temperature=1.0):
    """按 BEHAVIOUR 选动作，返回 (slot, action, old_logp)。"""
    uniq_slots = list(dict.fromkeys(slots))
    if FORCE_WIN:
        ws = [s for s, a in zip(slots, legal)
              if a.type in (E.ACT_TSUMO, E.ACT_RON)]
        if ws:
            pick = ws[0]
            return pick, resolve_action_for_slot(legal, slots, pick, logits_row), 0.0

    if BEHAVIOUR == "policy" and logits_row is not None:
        mask = torch.full((N_ACTIONS,), -1e9, device=logits_row.device)
        mask[uniq_slots] = 0.0
        scaled = (logits_row + mask) / max(float(temperature), 1e-6)
        logp_all = F.log_softmax(scaled, dim=-1)
        probs = logp_all.exp()
        pick = int(torch.multinomial(probs, 1).item())
        if pick not in uniq_slots:
            pick = uniq_slots[0]
        act = resolve_action_for_slot(legal, slots, pick, logits_row)
        return pick, act, float(logp_all[pick].item())

    if BEHAVIOUR == "shanten" and env_.phase == E.Phase.DISCARD:
        ts = [a for a in legal if a.type == E.ACT_TSUMO]
        if ts:
            return slot_of(ts[0]), ts[0], 0.0
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
            return best, resolve_action_for_slot(legal, slots, best), 0.0

    pick = random.choice(uniq_slots)
    old_lp = float(-np.log(max(1, len(uniq_slots))))
    return pick, resolve_action_for_slot(legal, slots, pick), old_lp


def _after_discard_shanten(env, p, act):
    if env.phase != E.Phase.DISCARD:
        return 8
    pl = env.players[p]
    if act.tile >= 0 and act.tile in pl.hand:
        pl.hand.remove(act.tile)
        pl.hand.sort()
        sa = hand_shanten(env, p)
        pl.hand.append(act.tile)
        pl.hand.sort()
        return sa
    return 8


def play_one_deal(model, hstate, seed, device, temperature, encoder,
                  shanten_weight, allow_ryuukyoku=False):
    """跑单局（向后兼容接口）。BEHAVIOUR='policy' 时用当前模型批量前向采样。"""
    results = play_deals_batched(
        model, [hstate], [seed], device, temperature, encoder,
        shanten_weight, allow_ryuukyoku=allow_ryuukyoku)
    return results[0]


def play_deals_batched(model, hstates, seeds, device, temperature, encoder,
                       shanten_weight, allow_ryuukyoku=False):
    """并行步进多个 deal，每步把全部待决策状态拼成一个 batch 送进模型前向。

    解决 CPU 上逐点 bs=1 前向过慢的问题，同时保证严格 on-policy 采样。
    """
    n_envs = len(hstates)
    envs = [h.new_env(seed=s) for h, s in zip(hstates, seeds)]
    infos = [[] for _ in range(n_envs)]
    results = [env.start_turn() for env in envs]
    guards = [0] * n_envs

    model.eval()
    while True:
        active = [i for i in range(n_envs)
                  if results[i] is None and guards[i] < 400]
        if not active:
            break

        # 收集所有活跃环境的待决策点
        pending_items = []   # (env_i, p, vec, sh, sit, give_up, tf, legal, slots)
        actions_by_env = {i: {} for i in active}
        for i in active:
            guards[i] += 1
            env = envs[i]
            hstate = hstates[i]
            for p in env.pending_players():
                raw_legal = env.legal_actions(p)
                legal = filter_legal_actions(raw_legal, allow_ryuukyoku)
                if not legal:
                    actions_by_env[i][p] = E.PASS_ACTION
                    continue
                vec, sh, sit, give_up, tf = encode(env, encoder, hstate, p)
                ev = ES.encode_event_stream(env, p, max_len=getattr(model, 'max_event_len', ES.DEFAULT_EVENT_LEN))
                slots = [slot_of(a) for a in legal]
                pending_items.append(
                    (i, p, vec, sh, sit, give_up, tf, legal, slots, ev))

        # 一次性批量前向推理
        logits_batch = None
        if pending_items and BEHAVIOUR == "policy":
            x_np = np.stack([it[2] for it in pending_items])
            tx_np = np.stack([it[6] for it in pending_items])
            ev_np = np.stack([it[9] for it in pending_items])
            with torch.no_grad():
                if isinstance(model, DualStreamMahjongNet):
                    logits_batch, _ = model(
                        torch.from_numpy(x_np).to(device),
                        torch.from_numpy(tx_np).to(device),
                        torch.from_numpy(ev_np).to(device),
                    )
                else:
                    logits_batch, _ = model(
                        torch.from_numpy(x_np).to(device),
                        torch.from_numpy(tx_np).to(device),
                    )

        # 采样动作并记录轨迹
        for idx, (i, p, vec, sh, sit, give_up, tf, legal, slots, ev) in enumerate(
                pending_items):
            lg_row = logits_batch[idx] if logits_batch is not None else None
            pick, act, old_lp = _pick_behaviour(
                envs[i], p, legal, slots,
                logits_row=lg_row, temperature=temperature)
            actions_by_env[i][p] = act
            sa = _after_discard_shanten(envs[i], p, act)
            infos[i].append(
                (vec, slots, pick, sh, sa, p, sit.label, give_up, tf, old_lp, ev))

        for i in active:
            results[i] = envs[i].step(actions_by_env[i])

    return [(envs[i], results[i], infos[i]) for i in range(n_envs)]


def _parse_aux_weights(text):
    """'tanyao=2,pinfu=2' -> {"tanyao": 2.0, "pinfu": 2.0}。"""
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


def _set_policy_trainable(model, trainable: bool):
    """Critic warmup 期间冻结主干与策略头，只训 value 头。"""
    for name, param in model.named_parameters():
        if name.startswith("value.") or name.startswith("value_head."):
            param.requires_grad = True
        else:
            param.requires_grad = trainable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hanchans", type=int, default=16, help="每次迭代并行跑几个半庄")
    ap.add_argument("--workers", type=int, default=1, help="并行环境子进程数（>1 时启用 Actor-Learner 多进程池）")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--entropy", type=float, default=0.005)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95,
                    help="GAE(lambda) 优势估计衰减系数（默认 0.95，替代高方差 MC 回报）")
    ap.add_argument("--uma-weight", type=float, default=1.0,
                    help="半庄终局顺位点（Uma）回传权重（默认 1.0，提供避四/保顺位激励）")
    ap.add_argument("--shanten-weight", type=float, default=0.0,
                    help="向听塑形权重（已修复为 sb - sa；从 BC 起步建议 0.0）")
    ap.add_argument("--deal-weight", type=float, default=0.0,
                    help="额外放铳惩罚权重（默认 0，r_point 已包含真实失分）")
    ap.add_argument("--survive-weight", type=float, default=0.0,
                    help="弃听保命奖权重（默认 0，避免鼓励消极弃和）")
    ap.add_argument("--place-weight", type=float, default=0.0,
                    help="顺位pt奖励权重（天凤 1位+30/2位+10/3位-10/4位-30，"
                         ">0 使 RL 目标对齐天凤排位而非纯点数最大化）")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--per-tile-head", action="store_true",
                    help="使用逐牌切牌头（保留 34 牌空间分辨率）")
    ap.add_argument("--arch", default="cnn", choices=["cnn", "dual_stream"])
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--event-len", type=int, default=ES.DEFAULT_EVENT_LEN)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--aux-bonus-weight", type=float, default=0.0,
                    help="额外任务奖励权重（>0 时把每局任务完成奖励加进 reward）")
    ap.add_argument("--aux-weights", default="",
                    help="风格任务权重，如 'tanyao=2,pinfu=2'（同一模型调参）")
    ap.add_argument("--init", default="",
                    help="从已有 ckpt 继续训练（自动读取模型超参，并默认启用 critic warmup 与 KL 锚定）")
    ap.add_argument("--critic-warmup", type=int, default=-1,
                    help="前 N 轮只训 value 头（-1 表示有 --init 时默认 10，否则 0）")
    ap.add_argument("--kl-coef", type=float, default=-1.0,
                    help="到初始 BC 策略的 KL 惩罚系数（-1 表示有 --init 时默认 0.02，否则 0）")
    ap.add_argument("--out", default="checkpoints/hanchan_net.pt")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--ppo-epochs", type=int, default=2,
                    help="每批 rollout 的 PPO 更新轮数")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--force-win", action="store_true")
    ap.add_argument("--allow-ryuukyoku", action="store_true",
                    help="允许听牌时主动宣言流局（默认关闭，防止奖励作弊）")
    ap.add_argument("--behaviour", default="policy",
                    choices=["policy", "random", "shanten"])
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    global FORCE_WIN, BEHAVIOUR
    FORCE_WIN = args.force_win
    BEHAVIOUR = args.behaviour
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    torch.manual_seed(args.seed0)
    random.seed(args.seed0)
    np.random.seed(args.seed0)

    ck = None
    in_planes = O2.N_PLANES
    aux_dim = 0
    per_tile_head = args.per_tile_head
    channels = args.channels
    blocks = args.blocks
    obs_order = False
    arch = args.arch
    d_model = args.d_model
    n_layers = args.n_layers
    n_heads = args.n_heads
    event_len = args.event_len
    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        if ck.get("arch") == "dual_stream_v1":
            arch = "dual_stream"
        d_model = ck.get("d_model", d_model)
        n_layers = ck.get("n_layers", n_layers)
        n_heads = ck.get("n_heads", n_heads)
        event_len = ck.get("event_len", event_len)
        channels = ck.get("channels", channels)
        blocks = ck.get("blocks", blocks)
        in_planes = ck.get("in_planes", in_planes)
        aux_dim = ck.get("aux_dim", 0)
        per_tile_head = ck.get("per_tile_head", per_tile_head)
        obs_order = bool(ck.get("obs_order", in_planes > O2.N_PLANES))

    if arch == "dual_stream":
        model = DualStreamMahjongNet(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            n_actions=N_ACTIONS, touch_dim=TOUCH.TOUCH_DIM,
            extra_scalar=SIT_DIM, in_planes=in_planes,
            aux_dim=aux_dim, max_event_len=event_len,
        ).to(device)
    else:
        # legacy ckpt: 从 head_fc.0.weight shape 反推 scalar_dim，
        # 使 load_state_dict 不因 366 vs 372 不匹配而崩溃
        scalar_dim = None
        if ck is not None and "state_dict" in ck:
            hf = ck["state_dict"].get("head_fc.0.weight")
            if hf is not None:
                in_scalar = hf.shape[1] - channels * 5
                scalar_dim = in_scalar - 8 - 12   # touch8 + sit12
                scalar_dim = max(scalar_dim, 0)
        model = MahjongNet(
            channels=channels, n_blocks=blocks,
            n_actions=N_ACTIONS,
            touch_dim=TOUCH.TOUCH_DIM,
            extra_scalar=SIT_DIM,
            in_planes=in_planes,
            aux_dim=aux_dim,
            per_tile_head=per_tile_head,
            scalar_dim=scalar_dim,
        ).to(device)

    ref_model = None
    if ck is not None:
        model.load_state_dict(ck["state_dict"], strict=False)
        print(f"从 {args.init} 继续训练"
              f"（iter={ck.get('iter')}, win={ck.get('win_rate')}）",
              flush=True)

    critic_warmup = (10 if args.init else 0) if args.critic_warmup < 0 else args.critic_warmup
    kl_coef = (0.02 if args.init else 0.0) if args.kl_coef < 0 else args.kl_coef
    if kl_coef > 0 and args.init:
        ref_model = copy.deepcopy(model).to(device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # legacy ckpt（in_planes=22）必须用旧 obs2（22平面/26标量）编码，
    # 否则喂 29 平面会让 forward shape 不匹配（610 vs 366）
    if in_planes <= 22:
        from mahjong import obs2_v1 as OENC
        print(f"[legacy obs] in_planes={in_planes} -> 使用 obs2_v1（22平面/26标量）", flush=True)
    else:
        OENC = O2
    encoder = OENC.ObsEncoder(order=obs_order)
    print(f"device={device} params={count_params(model):,} "
          f"hanchans/iter={args.hanchans} behaviour={BEHAVIOUR} "
          f"critic_warmup={critic_warmup} kl_coef={kl_coef} "
          f"per_tile_head={per_tile_head}", flush=True)

    t0 = time.time()
    tot_deals = tot_wins = 0
    give_up_seen = defense_seen = lastchance_seen = 0
    mp_pool = (
        ParallelDealPool(
            n_workers=args.workers,
            obs_order=obs_order,
            event_len=event_len,
            allow_ryuukyoku=args.allow_ryuukyoku,
        ).start()
        if args.workers > 1 and BEHAVIOUR == "policy"
        else None
    )

    def _pool_policy_fn(big_v, big_tf, big_ev, slot_lists):
        model.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(big_v).to(device)
            tx_t = torch.from_numpy(big_tf).to(device)
            if isinstance(model, DualStreamMahjongNet):
                ev_t = torch.from_numpy(big_ev).to(device)
                lg_b, _ = model(x_t, tx_t, ev_t)
            else:
                lg_b, _ = model(x_t, tx_t)
        out = []
        temp = max(float(args.temperature), 1e-6)
        for row_i, slots in enumerate(slot_lists):
            uniq = list(dict.fromkeys(slots))
            if FORCE_WIN:
                for ws in (SLOT_TSUMO, SLOT_RON):
                    if ws in uniq:
                        out.append((ws, 0.0, -1))
                        break
                else:
                    pass
                if len(out) == row_i + 1:
                    continue
            mask = torch.full((N_ACTIONS,), -1e9, device=device)
            mask[uniq] = 0.0
            logp = F.log_softmax((lg_b[row_i] + mask) / temp, dim=-1)
            pick = int(torch.multinomial(logp.exp(), 1).item())
            if pick not in uniq:
                pick = uniq[0]
            best_rt = int(lg_b[row_i, :34].argmax().item()) if pick == SLOT_RIICHI else -1
            out.append((pick, float(logp[pick].item()), best_rt))
        return out

    for it in range(args.iters):
        in_warmup = it < critic_warmup
        _set_policy_trainable(model, not in_warmup)

        # 按 (半庄, 玩家) 收集完整轨迹，便于半庄终局 Uma 回传与 GAE(lambda) 优势估计
        hc_trajs = [{p: [] for p in range(4)} for _ in range(args.hanchans)]
        hstates = [HanchanState(dealer=random.randrange(4), start_points=25000)
                   for _ in range(args.hanchans)]
        deal_counters = [args.seed0 * 104729 + it * 257 + hc
                         for hc in range(args.hanchans)]

        # 并行推进所有未结束的半庄
        while True:
            alive_idx = [i for i, h in enumerate(hstates) if not h.finished]
            if not alive_idx:
                break
            batch_h = [hstates[i] for i in alive_idx]
            batch_s = [deal_counters[i] for i in alive_idx]
            for i in alive_idx:
                deal_counters[i] += 1

            if mp_pool is not None:
                deal_outs = mp_pool.run_deals(batch_h, batch_s, _pool_policy_fn)
            else:
                deal_outs = play_deals_batched(
                    model, batch_h, batch_s, device, args.temperature,
                    encoder, args.shanten_weight,
                    allow_ryuukyoku=args.allow_ryuukyoku)

            for idx_in_batch, i in enumerate(alive_idx):
                env, result, info = deal_outs[idx_in_batch]
                hstate = hstates[i]
                if result is None:
                    hstate.finished = True
                    continue
                tot_deals += 1
                if result.result_type != "ryuukyoku":
                    tot_wins += 1
                for rec in info:
                    label, gu = rec[6], rec[7]
                    if label == "defense":
                        defense_seen += 1
                    if label == "last_chance":
                        lastchance_seen += 1
                    if gu:
                        give_up_seen += 1

                # 使用含立直棒成本的真实净点差 net_points
                net_deltas = E.net_points(result, env.initial_points)
                by_player = {}
                for rec in info:
                    by_player.setdefault(rec[5], []).append(rec)
                for p, lst in by_player.items():
                    r_term = SC.reward_for_result(
                        p, result, _sit_stub(lst[-1][6]),
                        hstate, give_up_flag=lst[-1][7],
                        w_deal=args.deal_weight,
                        w_survive=args.survive_weight,
                        w_place=args.place_weight)
                    # 补齐立直 1000 点真实成本差
                    r_term += (net_deltas[p] - result.points_delta[p]) / 10000.0
                    if args.aux_bonus_weight > 0:
                        _w = _parse_aux_weights(args.aux_weights)
                        _, _aux = AUX.deal_bonus(env, result, p, weights=_w)
                        r_term += args.aux_bonus_weight * _aux / 1000.0

                    n = len(lst)
                    for k, (vec, legal_slots, pick, sb, sa, _, label, gu, tf, old_lp, ev) \
                            in enumerate(lst):
                        step_r = 0.0
                        if args.shanten_weight > 0 and sb < 8 and sa < 8:
                            step_r += args.shanten_weight * (sb - sa)
                        if k == n - 1:
                            step_r += r_term
                        done_flag = (k == n - 1)
                        hc_trajs[i][p].append((vec, legal_slots, pick, tf, old_lp, ev, step_r, done_flag))
                hstate.advance(result)

        # 半庄结束后回传终局顺位点（Uma）到该玩家在本半庄最后一步
        vecs, slots, touches, old_lps, evs, step_rewards, dones = [], [], [], [], [], [], []
        for i, hstate in enumerate(hstates):
            uma_map = placement_bonus(hstate.final_placement(), hanchan=not hstate.east_only)
            for p in range(4):
                traj = hc_trajs[i][p]
                if not traj:
                    continue
                uma_r = args.uma_weight * (uma_map[p] / 30000.0)
                for k, (vec, legal_slots, pick, tf, old_lp, ev, sr, dn) in enumerate(traj):
                    if k == len(traj) - 1:
                        sr = sr + uma_r
                        dn = True
                    else:
                        dn = False
                    vecs.append(vec)
                    slots.append((pick, legal_slots))
                    touches.append(tf)
                    old_lps.append(old_lp)
                    evs.append(ev)
                    step_rewards.append(sr)
                    dones.append(dn)

        if not vecs:
            continue

        old_lp_t = torch.tensor(old_lps, dtype=torch.float32, device=device)

        CH = 2048
        raw_chunks = []
        for start in range(0, len(vecs), CH):
            sl = slots[start:start + CH]
            x = torch.from_numpy(np.stack(vecs[start:start + CH])).to(device)
            tx = torch.from_numpy(np.stack(touches[start:start + CH])).to(device)
            evx = torch.from_numpy(np.stack(evs[start:start + CH])).to(device)
            old_lp_chunk = old_lp_t[start:start + CH]
            bsz = len(sl)
            mrow = torch.full((bsz, N_ACTIONS), -1e9, device=device)
            for idx_row, (_, legal_slots) in enumerate(sl):
                for s in legal_slots:
                    mrow[idx_row, s] = 0.0
            pick_t = torch.tensor([p for (p, _) in sl], dtype=torch.long, device=device)
            raw_chunks.append((x, tx, evx, mrow, pick_t, old_lp_chunk))

        # 用当前 Critic 计算 V(s) 与 GAE(lambda) 优势及回报目标
        model.eval()
        all_vals = []
        ref_logps = []
        with torch.no_grad():
            for x, tx, evx, mrow, pick_t, _ in raw_chunks:
                _, values = (model(x, tx, evx) if isinstance(model, DualStreamMahjongNet) else model(x, tx))
                all_vals.append(values.cpu().numpy())
                if ref_model is not None:
                    r_logits, _ = (ref_model(x, tx, evx) if isinstance(ref_model, DualStreamMahjongNet) else ref_model(x, tx))
                    ref_logps.append(F.log_softmax(r_logits + mrow, dim=-1))
        vals_np = np.concatenate(all_vals, axis=0)
        n_total = len(step_rewards)
        gae_adv = np.zeros(n_total, dtype=np.float32)
        gae_ret = np.zeros(n_total, dtype=np.float32)
        gae_running = 0.0
        lam = float(args.gae_lambda)
        for idx_t in range(n_total - 1, -1, -1):
            next_v = 0.0 if dones[idx_t] or idx_t + 1 >= n_total else float(vals_np[idx_t + 1])
            mask_nonterm = 0.0 if dones[idx_t] else 1.0
            delta_t = float(step_rewards[idx_t]) + args.gamma * next_v * mask_nonterm - float(vals_np[idx_t])
            gae_running = delta_t + args.gamma * lam * mask_nonterm * gae_running
            gae_adv[idx_t] = gae_running
            gae_ret[idx_t] = gae_running + float(vals_np[idx_t])

        rt = torch.from_numpy(gae_ret).to(device)
        adv_all = torch.from_numpy(gae_adv).to(device)
        adv_all = (adv_all - adv_all.mean()) / adv_all.std().clamp_min(1e-6)
        chunks = []
        adv_chunks = []
        for ci, (x, tx, evx, mrow, pick_t, old_lp_chunk) in enumerate(raw_chunks):
            start = ci * CH
            end = start + len(pick_t)
            chunks.append((x, tx, evx, rt[start:end], mrow, pick_t, old_lp_chunk))
            adv_chunks.append(adv_all[start:end])

        model.train()
        _pol_acc = _v_acc = _ent_acc = _kl_acc = 0.0
        _nb = 0
        n_chunks = max(1, len(chunks))
        ppo_epochs = max(1, int(args.ppo_epochs))
        clip = float(args.clip)

        for _ep in range(ppo_epochs):
            opt.zero_grad()
            for ci, (x, tx, evx, rt_chunk, mrow, pick_t, old_lp_chunk) in enumerate(chunks):
                logits, values = (model(x, tx, evx) if isinstance(model, DualStreamMahjongNet) else model(x, tx))
                vloss = F.mse_loss(values, rt_chunk)

                if in_warmup:
                    loss = vloss / n_chunks
                    loss.backward()
                    _v_acc += vloss.item()
                    _nb += 1
                    continue

                logp_all = F.log_softmax(logits + mrow, dim=-1)
                logp_pick = logp_all.gather(1, pick_t.unsqueeze(1)).squeeze(1)
                probs_all = logp_all.exp()
                ent = -(probs_all * logp_all).sum(-1).mean()
                adv_chunk = adv_chunks[ci]

                ratio = torch.exp((logp_pick - old_lp_chunk).clamp(-10.0, 10.0))
                clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
                pol = -torch.min(ratio * adv_chunk, clipped * adv_chunk).mean()

                kl_loss = torch.tensor(0.0, device=device)
                if ref_model is not None and kl_coef > 0:
                    kl_loss = (probs_all * (logp_all - ref_logps[ci])).sum(-1).mean()

                loss = (pol + 0.5 * vloss - args.entropy * ent + kl_coef * kl_loss) / n_chunks
                loss.backward()
                _pol_acc += pol.item()
                _v_acc += vloss.item()
                _ent_acc += ent.item()
                _kl_acc += kl_loss.item()
                _nb += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()

        if args.save_every and (it + 1) % args.save_every == 0:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save({
                "arch": "dual_stream_v1" if arch == "dual_stream" else "hanchan_ppo",
                "d_model": int(d_model),
                "n_layers": int(n_layers),
                "n_heads": int(n_heads),
                "event_len": int(event_len),
                "state_dict": model.state_dict(),
                "channels": channels,
                "blocks": blocks,
                "in_planes": int(in_planes),
                "obs_order": bool(obs_order),
                "aux_dim": int(aux_dim),
                "per_tile_head": bool(per_tile_head),
                "touch_dim": TOUCH.TOUCH_DIM,
                "sit_dim": SIT_DIM,
                "extra_scalar": SIT_DIM,
                "iter": it + 1,
                "deals": tot_deals,
                "win_rate": 100.0 * tot_wins / max(1, tot_deals),
            }, args.out.replace(".pt", "_it%d.pt" % (it + 1)))

        if it % args.log_every == 0 or it == args.iters - 1:
            dt = time.time() - t0
            wr = 100.0 * tot_wins / max(1, tot_deals)
            nb = max(1, _nb)
            tag = "[warmup]" if in_warmup else ""
            print(f"iter {it:4d}{tag} deals={tot_deals:5d} win={wr:5.2f}% "
                  f"R={rt.mean().item():+.3f} "
                  f"pol={_pol_acc / nb:+.4f} vloss={_v_acc / nb:.4f} "
                  f"ent={_ent_acc / nb:.3f} kl={_kl_acc / nb:.4f} "
                  f"giveup={give_up_seen} defend={defense_seen} "
                  f"last={lastchance_seen} {dt:.0f}s", flush=True)

    if mp_pool is not None:
        mp_pool.close()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "arch": "dual_stream_v1" if arch == "dual_stream" else "hanchan_ppo",
        "d_model": int(d_model),
        "n_layers": int(n_layers),
        "n_heads": int(n_heads),
        "event_len": int(event_len),
        "state_dict": model.state_dict(),
        "channels": channels,
        "blocks": blocks,
        "in_planes": int(in_planes),
        "obs_order": bool(obs_order),
        "aux_dim": int(aux_dim),
        "per_tile_head": bool(per_tile_head),
        "touch_dim": TOUCH.TOUCH_DIM,
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
