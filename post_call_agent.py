"""方案 1A：Post-Call State Value（叫后状态前瞻）副露决策器。

核心思路（解决「外挂叫牌头 + 门清主模型」10 条路径全负收益）：
1. 响应阶段合法叫牌动作极少（最多 1 个 PASS + 1..4 个吃/碰/明杠）；
2. 对每个合法叫牌动作在轻量环境副本里试走一步，并选出叫后最优切牌，
   得到真实的「叫后 13 张局面 s'_call」；
3. 直接用主模型的 Value 塔比较 V(s'_call) 与 V(s_pass)，并结合保役门控
   （役牌 / 断幺九 / 混一色 / 对对和 / 终盘形式听牌流听）与向听收益过滤，
   彻底消除「叫完变成无役死牌」的负收益。

用法（对比评测 PostCallValueAgent vs 纯主模型 no_call）：
    python post_call_agent.py --ckpt checkpoints/dual_bc_v2.pt --deals 120 --margin 0.02
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mahjong import env as E
from mahjong import event_stream as ES
from mahjong import fast_cache as FC
from mahjong import obs2 as O2
from mahjong import tiles as T
from mahjong.hanchan import HanchanState


def has_open_yaku_potential(env, p_idx: int) -> tuple[bool, str]:
    """判断副露后手牌是否具备可成立的副露役或终盘流听价值。

    返回 (ok, reason)：
    - yakuhai: 已有白/发/中、场风或自风刻子/杠
    - tanyao: 已有副露全部为中张(2..8)，且暗牌中幺九字牌 <= 2 张（可清干净成断幺九）
    - honitsu: 已有副露与绝大部分暗牌集中在单花色+字牌（杂牌 <= 2 张）
    - toitoi: 对子/刻子数 >= 4（含副露刻子）
    - keiten: 终盘剩余可摸牌 <= 16 张且叫后已听牌（抢荒牌流局形式听牌 +1500/+3000）
    """
    p = env.players[p_idx]
    yakuhai_tiles = {T.HAKU, T.HATSU, T.CHUN, p.seat_wind, env.round_wind}

    # 1. 已有役牌刻子/杠（明暗皆可）
    for kind, t, _ in p.melds:
        if kind != "chi" and t in yakuhai_tiles:
            return True, "yakuhai"
    counts = p.counts()
    for t in yakuhai_tiles:
        if 0 <= t < T.NUM_TILES and counts[t] >= 3:
            return True, "yakuhai"

    # 2. 断幺九潜力：所有副露不含幺九字牌，且暗牌中幺九字牌 <= 2 张
    melds_tanyao_ok = True
    for kind, t, _ in p.melds:
        tiles = (t, t + 1, t + 2) if kind == "chi" else (t,)
        if any(T.is_yao9(x) for x in tiles):
            melds_tanyao_ok = False
            break
    if melds_tanyao_ok:
        yao9_in_hand = sum(1 for x in p.hand if T.is_yao9(x))
        if yao9_in_hand <= 2:
            return True, "tanyao"

    # 3. 混一色/清一色潜力：副露不超过 1 种数牌花色，且暗牌中异花色数牌 <= 2 张
    meld_suits = set()
    for kind, t, _ in p.melds:
        if not T.is_honor(t):
            meld_suits.add(T.suit_of(t))
    if len(meld_suits) <= 1:
        for target_suit in (meld_suits if meld_suits else (0, 1, 2)):
            off_suit = sum(
                1 for x in p.hand
                if (not T.is_honor(x)) and T.suit_of(x) != target_suit
            )
            if off_suit <= 2:
                return True, "honitsu"

    # 4. 对对和潜力：无顺子副露且暗牌+副露刻对 >= 4
    if all(kind != "chi" for kind, _, _ in p.melds):
        pair_or_trip = sum(1 for c in counts if c >= 2)
        if len(p.melds) + pair_or_trip >= 4:
            return True, "toitoi"

    # 5. 终盘形式听牌（流听）：剩余可摸牌 <= 16 且叫完即听牌
    dead = getattr(env, "dead_wall_size", 0)
    wall_left = max(0, len(env.wall) - dead - env.wall_idx)
    if wall_left <= 16 and env._is_tenpai(p_idx):
        return True, "keiten"

    return False, "no_yaku"


def _norm_counts_for_shanten(p) -> tuple[int, ...]:
    c = p.full_counts()
    for kind, t, _ in p.melds:
        if kind in ("ankan", "minkan", "addkan") and c[t] > 0:
            c[t] -= 1
    return tuple(c)


def simulate_post_call_state(env, p_idx: int, act: E.Action, discard_selector=None):
    """在轻量副本里执行叫牌动作并切掉最优一张，返回 (sim_env, chosen_discard_tile, sh_after)。

    不修改原始 env。
    """
    sim = copy.deepcopy(env)
    tile = sim.last_discard
    if act.type == E.ACT_CHI:
        sim._do_chi(p_idx, act, tile)
    elif act.type == E.ACT_PON:
        sim._do_pon(p_idx, tile)
    elif act.type == E.ACT_MINKAN:
        sim._do_minkan_from_discard(p_idx, E.ACT_MINKAN, tile)
    else:
        return None, -1, 8

    # 叫牌后进入 DISCARD 阶段，模拟切出一张牌得到 13 张（归一化）局面
    p = sim.players[p_idx]
    legal_disc = [a for a in sim.legal_actions(p_idx) if a.type == E.ACT_DISCARD]
    if not legal_disc:
        return sim, -1, FC.cached_shanten(_norm_counts_for_shanten(p))

    if discard_selector is not None:
        best_tile = discard_selector(sim, p_idx, legal_disc)
    else:
        # 默认快速前瞻：选打完后向听数最小（平手取进张最多）的弃牌
        best_tile = legal_disc[0].tile
        best_key = (99, 0)
        for da in legal_disc:
            t = da.tile
            if t not in p.hand:
                continue
            p.hand.remove(t)
            p.hand.sort()
            c_norm = _norm_counts_for_shanten(p)
            sh = FC.cached_shanten(c_norm)
            w = FC.fast_wait_count(c_norm) if sh == 0 else 0
            p.hand.append(t)
            p.hand.sort()
            key = (sh, -w)
            if key < best_key:
                best_key = key
                best_tile = t

    if best_tile in p.hand:
        p.hand.remove(best_tile)
        p.hand.sort()
        p.river.append(best_tile)
        p.genbutsu.add(best_tile)
    sh_after = FC.cached_shanten(_norm_counts_for_shanten(p))
    return sim, best_tile, sh_after


class PostCallValueAgent:
    """用主模型 Value 塔做叫后状态前瞻（Post-Call State Value）的副露 Agent。

    支持 MahjongNet 与 DualStreamMahjongNet。
    """

    def __init__(
        self,
        ckpt_path: str,
        margin: float = 0.02,
        require_yaku_potential: bool = True,
        require_shanten_gain: bool = True,
        device: str = "cpu",
    ):
        import torch
        from mahjong.dual_net import DualStreamMahjongNet
        from mahjong.net import MahjongNet
        from mahjong.touch import TOUCH_DIM
        from train_hanchan import N_ACTIONS, SIT_DIM

        self.torch = torch
        self.device = device
        self.margin = float(margin)
        self.require_yaku_potential = bool(require_yaku_potential)
        self.require_shanten_gain = bool(require_shanten_gain)
        self.hstate = None
        self.calls = 0
        self.call_reasons = {}

        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.is_dual = ck.get("arch") == "dual_stream_v1"
        in_planes = int(ck.get("in_planes", O2.N_PLANES))
        obs_order = bool(ck.get("obs_order", in_planes > O2.N_PLANES))
        self.enc = O2.ObsEncoder(order=obs_order)
        self.event_len = int(ck.get("event_len", ES.DEFAULT_EVENT_LEN))

        if self.is_dual:
            self.model = DualStreamMahjongNet(
                d_model=int(ck.get("d_model", 128)),
                n_layers=int(ck.get("n_layers", 3)),
                n_heads=int(ck.get("n_heads", 4)),
                n_actions=N_ACTIONS,
                touch_dim=int(ck.get("touch_dim", TOUCH_DIM)),
                extra_scalar=int(ck.get("extra_scalar", SIT_DIM)),
                in_planes=in_planes,
                aux_dim=int(ck.get("aux_dim", 0)),
                max_event_len=self.event_len,
            ).to(device)
        else:
            self.model = MahjongNet(
                channels=int(ck.get("channels", 64)),
                n_blocks=int(ck.get("blocks", 4)),
                n_actions=N_ACTIONS,
                touch_dim=int(ck.get("touch_dim", TOUCH_DIM)),
                extra_scalar=int(ck.get("extra_scalar", SIT_DIM)),
                in_planes=in_planes,
                aux_dim=int(ck.get("aux_dim", 0)),
                per_tile_head=bool(ck.get("per_tile_head", False)),
            ).to(device)
        self.model.load_state_dict(ck["state_dict"], strict=False)
        self.model.eval()

    def set_game(self, env, hstate=None):
        self.hstate = hstate

    def _infer(self, env, p_idx: int):
        from train_hanchan import encode

        hstate = self.hstate or HanchanState(dealer=env.dealer)
        vec, sh, sit, gu, tf = encode(env, self.enc, hstate, p_idx)
        x_t = self.torch.from_numpy(vec).unsqueeze(0).to(self.device)
        tf_t = self.torch.from_numpy(tf).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            if self.is_dual:
                ev = ES.encode_event_stream(env, p_idx, max_len=self.event_len)
                ev_t = self.torch.from_numpy(ev).unsqueeze(0).to(self.device)
                logits, val = self.model(x_t, tf_t, ev_t)
            else:
                logits, val = self.model(x_t, tf_t)
        return logits[0], float(val[0].item()), sh

    def _model_discard_selector(self, sim_env, p_idx: int, legal_disc):
        logits, _, _ = self._infer(sim_env, p_idx)
        best_a = max(legal_disc, key=lambda a: float(logits[a.tile].item()))
        return best_a.tile

    def act(self, env, p_idx: int, legal=None) -> E.Action:
        from train_hanchan import filter_legal_actions, resolve_action_for_slot, slot_of

        raw_legal = legal if legal is not None else env.legal_actions(p_idx)
        legal = filter_legal_actions(raw_legal, allow_ryuukyoku=False)
        if not legal:
            return E.PASS_ACTION

        # 必和优先
        for a in legal:
            if a.type in (E.ACT_TSUMO, E.ACT_RON):
                return a

        logits, v_base, sh_base = self._infer(env, p_idx)

        # 响应阶段：用 Post-Call State Value 比较所有合法叫牌 vs PASS
        if env.phase == E.Phase.RESPONSE:
            call_acts = [a for a in legal if a.type in (E.ACT_PON, E.ACT_CHI, E.ACT_MINKAN)]
            if not call_acts:
                return E.PASS_ACTION

            best_act = E.PASS_ACTION
            best_gain = self.margin
            best_reason = ""

            for ca in call_acts:
                sim_env, _, sh_after = simulate_post_call_state(
                    env, p_idx, ca, discard_selector=self._model_discard_selector
                )
                if sim_env is None:
                    continue
                yaku_ok, reason = has_open_yaku_potential(sim_env, p_idx)
                if self.require_yaku_potential and not yaku_ok:
                    continue
                if self.require_shanten_gain and (sh_after >= sh_base) and reason != "keiten":
                    continue
                _, v_after, _ = self._infer(sim_env, p_idx)
                gain = v_after - v_base
                if gain > best_gain:
                    best_gain = gain
                    best_act = ca
                    best_reason = reason

            if best_act.type != E.ACT_PASS:
                self.calls += 1
                self.call_reasons[best_reason] = self.call_reasons.get(best_reason, 0) + 1
            return best_act

        # 弃牌阶段：走主模型合法槽 argmax
        slots = [slot_of(a) for a in legal]
        uniq = list(dict.fromkeys(slots))
        best_slot = max(uniq, key=lambda s: float(logits[s].item()))
        return resolve_action_for_slot(legal, slots, best_slot, logits)


def main():
    ap = argparse.ArgumentParser(description="Post-Call State Value 副露决策同 seed 对照评测")
    ap.add_argument("--ckpt", required=True, help="主模型 checkpoint（需含训练好的 Value 塔）")
    ap.add_argument("--deals", type=int, default=120, help="评测局数")
    ap.add_argument("--margin", type=float, default=0.02, help="V(call) - V(pass) 最小正收益阈值")
    ap.add_argument("--seed0", type=int, default=42000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    agents = [PostCallValueAgent(args.ckpt, margin=args.margin, device=args.device) for _ in range(4)]
    wins = open_wins = dealins = draws = tenpai_on_draw = 0
    for d in range(args.deals):
        h = HanchanState(dealer=d % 4)
        env = h.new_env(seed=args.seed0 + d)
        for ag in agents:
            ag.set_game(env, h)
        res = env.start_turn()
        guard = 0
        while res is None and guard < 400:
            guard += 1
            acts = {p: agents[p].act(env, p) for p in env.pending_players()}
            res = env.step(acts)
        if res is None:
            continue
        if res.result_type == "ryuukyoku":
            draws += 1
            tenpai_on_draw += sum(1 for x in res.tenpai if x)
        else:
            wins += 1
            if len(env.players[res.winner].melds) > 0:
                open_wins += 1
            if res.result_type == "ron":
                dealins += 1

    seat_den = max(1, args.deals * 4)
    tot_calls = sum(ag.calls for ag in agents)
    print(
        f"deals={args.deals}  calls={tot_calls} ({tot_calls / max(1, args.deals):.2f}/局)  "
        f"seat_win={100.0 * wins / seat_den:.2f}%  open_wins={open_wins}/{wins}  "
        f"seat_dealin={100.0 * dealins / seat_den:.2f}%  "
        f"draw_tenpai={100.0 * tenpai_on_draw / max(1, draws * 4):.2f}%"
    )


if __name__ == "__main__":
    main()
