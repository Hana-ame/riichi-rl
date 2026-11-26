"""非 ML 日麻策略 · 第 2 版（v2）：两轮迭代 + 三家做牌推测与放铳防守（门清攻防版）。

相比第 1 版的升级（v2 / 共 4 版）：
- 新增 `OpponentInference`（纯规则与概率推理，零 ML）：
  1. 推测三家听牌概率 P_tenpai(q) 与高危状态（立直 / 大副露）；
  2. 推测三家做牌方向（染手/混一色花色亲和、对对和、断幺、役牌/宝牌偏好），
     生成持牌概率 P_hold(q, t) 并据此**修正牌山摸牌概率 P(draw t)**；
  3. 结合现物（genbutsu）、筋（suji）、壁（kabe）、早巡外侧、字牌生张，
     计算每张牌的三家综合放铳概率与期望失点 C_deal_in(t)。
- 引入统一攻防点数期望：面临立直/大副露高危威胁时，自动允许退向听兜牌或弃和。
- 仍保持门清优先（不主动吃碰），便于对比「加入三家推测与防守」带来的纯收益。
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from mahjong import env as E
from mahjong import policies as POL
from mahjong import refengine as REF
from mahjong import tiles as T
from mahjong.tiles import NUM_TILES


def _public_visible_counts(env, p_idx: int) -> List[int]:
    vis = [0] * NUM_TILES
    for d in getattr(env, "dora_indicators", ()):
        if 0 <= d < NUM_TILES:
            vis[d] += 1
    for q in env.players:
        for t in q.river:
            vis[t] += 1
        for kind, t, _ in q.melds:
            if kind == "chi":
                vis[t] += 1
                vis[t + 1] += 1
                vis[t + 2] += 1
            elif kind == "pon":
                vis[t] += 3
            else:
                vis[t] += 4
    for t in env.players[p_idx].hand:
        vis[t] += 1
    return vis


def _dora_tiles(env) -> List[int]:
    return [T.dora_from_indicator(d) for d in getattr(env, "dora_indicators", ())]


def _live_wall_left(env) -> int:
    total = len(getattr(env, "wall", ()))
    dead = getattr(env, "dead_wall_size", 14)
    idx = getattr(env, "wall_idx", 0)
    return max(1, total - dead - idx)


def _shanten_cached(counts: Sequence[int], cache: Dict[Tuple[int, ...], int]) -> int:
    key = tuple(counts)
    val = cache.get(key)
    if val is None:
        val = REF.ref_shanten(counts)
        cache[key] = val
    return val


class OpponentInference:
    """对其他三家手牌方向、听牌概率、持牌概率与放铳危险度的非 ML 推测。"""

    def __init__(self, env, me_idx: int, vis: Sequence[int], unseen: Sequence[int]):
        self.vis = vis
        self.unseen = unseen
        self.doras = set(_dora_tiles(env))
        self.round_wind = getattr(env, "round_wind", T.EAST)

        self.tenpai_prob: Dict[int, float] = {}
        self.is_high_threat: Dict[int, bool] = {}
        self.hold_prob: Dict[int, List[float]] = {}
        self.ron_prob: Dict[int, List[float]] = {}
        self.est_value: Dict[int, float] = {}

        for q in env.players:
            if q.idx != me_idx:
                self._infer_opponent(q)

    def _infer_opponent(self, q) -> None:
        river, melds, genbutsu = q.river, q.melds, q.genbutsu
        n_turn, n_melds = len(river), len(melds)

        if q.riichi:
            p_tenpai, high_threat = 1.0, True
        else:
            base = 1.0 / (1.0 + math.exp(-0.36 * (n_turn - 11.5)))
            meld_boost = (0.0, 0.08, 0.28, 0.58, 0.88)[min(4, n_melds)]
            late_simples = sum(1 for t in river[max(0, n_turn - 4):] if T.is_simple(t))
            p_tenpai = min(0.90, base * 0.65 + meld_boost + (0.025 * late_simples if n_turn >= 7 else 0.0))
            high_threat = (n_melds >= 2 and n_turn >= 6) or (n_melds >= 3)
        self.tenpai_prob[q.idx] = p_tenpai
        self.is_high_threat[q.idx] = high_threat

        suit_discards = [0, 0, 0, 0]
        for t in river:
            suit_discards[T.suit_of(t)] += 1
        suit_melds = [0, 0, 0, 0]
        pon_count = 0
        has_yaochu_meld = False
        for kind, t, _ in melds:
            suit_melds[T.suit_of(t)] += 1
            if kind in ("pon", "minkan", "ankan", "addkan"):
                pon_count += 1
            if (kind == "chi" and (T.is_yao9(t) or T.is_yao9(t + 2))) or (kind != "chi" and T.is_yao9(t)):
                has_yaochu_meld = True

        aff = [1.0, 1.0, 1.0, 1.0]
        flush_suit = -1
        for s in range(3):
            other_disc = sum(suit_discards[o] for o in range(3) if o != s)
            other_melds = sum(suit_melds[o] for o in range(3) if o != s)
            if other_melds == 0 and (suit_melds[s] >= 1 or (n_turn >= 5 and suit_discards[s] <= 1 and other_disc >= 4)):
                aff[s] = min(2.3, 1.0 + 0.35 * suit_melds[s] + 0.10 * max(0, other_disc - 3))
                if aff[s] >= 1.5:
                    flush_suit = s
            elif suit_discards[s] >= 4 and suit_melds[s] == 0:
                aff[s] = 0.72
        if flush_suit >= 0:
            for o in range(3):
                if o != flush_suit:
                    aff[o] *= 0.55
            if suit_discards[3] <= 1:
                aff[3] = 1.35

        yaochu_disc = sum(1 for t in river if T.is_yao9(t))
        tanyao_like = (not has_yaochu_meld) and (n_melds >= 1 or yaochu_disc >= max(3, n_turn // 2))
        toitoi_like = pon_count >= 2

        val = 6500.0 if q.riichi else 3600.0
        if flush_suit >= 0:
            val += 2400.0
        if toitoi_like:
            val += 1500.0
        for kind, t, _ in melds:
            if kind == "chi":
                val += 1600.0 * sum(1 for x in (t, t + 1, t + 2) if x in self.doras)
            else:
                if t in self.doras:
                    val += 4800.0
                if t in (q.seat_wind, self.round_wind, T.HAKU, T.HATSU, T.CHUN):
                    val += 1000.0
        if q.is_dealer:
            val *= 1.45
        self.est_value[q.idx] = val

        early_river = set(river[:min(4, n_turn)])
        yakuhai = {q.seat_wind, self.round_wind, T.HAKU, T.HATSU, T.CHUN}
        holds = [0.0] * NUM_TILES
        rons = [0.0] * NUM_TILES

        for t in range(NUM_TILES):
            u = self.unseen[t]
            if u <= 0:
                continue
            s = T.suit_of(t)
            h = min(0.65, (u / 4.0) * (13.0 / 90.0) * aff[s])
            if t in genbutsu:
                h *= 0.25
            if t in self.doras:
                h *= 1.35
            if t in yakuhai and t not in genbutsu:
                h *= 1.25
            holds[t] = min(0.75, h)

            if t in genbutsu:
                rons[t] = 0.0
                continue

            if T.is_honor(t):
                r_p = 0.015 if u == 1 else (0.045 if u == 2 else 0.085)
                if t in yakuhai:
                    r_p *= 1.45
                if toitoi_like:
                    r_p *= 1.35
                if tanyao_like and not q.riichi:
                    r_p *= 0.45
                r_p *= aff[3]
            else:
                rank = t % 9 + 1
                base_idx = s * 9
                r_p = (0.0, 0.055, 0.085, 0.115, 0.145, 0.155, 0.145, 0.115, 0.085, 0.055)[rank]
                suji_down = (rank >= 4) and ((base_idx + rank - 4) in genbutsu)
                suji_up = (rank <= 6) and ((base_idx + rank + 2) in genbutsu)
                if rank in (1, 2, 3) and suji_up:
                    r_p *= 0.40
                elif rank in (7, 8, 9) and suji_down:
                    r_p *= 0.40
                elif rank in (4, 5, 6):
                    r_p *= 0.30 if (suji_down and suji_up) else (0.66 if (suji_down or suji_up) else 1.0)

                if rank == 1 and self.unseen[base_idx + 1] == 0:
                    r_p *= 0.32
                elif rank == 2 and self.unseen[base_idx + 2] == 0:
                    r_p *= 0.42
                elif rank == 9 and self.unseen[base_idx + 7] == 0:
                    r_p *= 0.32
                elif rank == 8 and self.unseen[base_idx + 6] == 0:
                    r_p *= 0.42
                elif 3 <= rank <= 7:
                    ld, rd = self.unseen[t - 1] == 0, self.unseen[t + 1] == 0
                    r_p *= 0.32 if (ld and rd) else (0.72 if (ld or rd) else 1.0)

                if rank in (1, 2) and (base_idx + 2) in early_river:
                    r_p *= 0.65
                if rank in (8, 9) and (base_idx + 6) in early_river:
                    r_p *= 0.65
                if tanyao_like and T.is_terminal(t) and not q.riichi:
                    r_p *= 0.45
                if toitoi_like and u >= 2:
                    r_p *= 1.25
                r_p *= aff[s]

            if t in self.doras:
                r_p *= 1.30
            rons[t] = min(0.45, r_p)

        self.hold_prob[q.idx] = holds
        self.ron_prob[q.idx] = rons

    def adjusted_draw_probs(self, wall_hold_weight: float = 0.25) -> List[float]:
        eff = [0.0] * NUM_TILES
        total = 0.0
        for t in range(NUM_TILES):
            u = self.unseen[t]
            if u <= 0:
                continue
            opp_hold_sum = sum(self.hold_prob[q][t] for q in self.hold_prob)
            w = u * max(0.35, min(1.15, 1.0 - wall_hold_weight * opp_hold_sum))
            eff[t] = w
            total += w
        return [w / total for w in eff] if total > 0.0 else [0.0] * NUM_TILES

    def deal_in_risk(self, tile: int) -> Tuple[float, float, float]:
        p_safe, loss_all, loss_high = 1.0, 0.0, 0.0
        for q_idx, p_tp in self.tenpai_prob.items():
            p_hit = p_tp * self.ron_prob[q_idx][tile]
            p_safe *= (1.0 - p_hit)
            loss = p_hit * self.est_value[q_idx]
            loss_all += loss
            if self.is_high_threat[q_idx]:
                loss_high += loss
        return 1.0 - p_safe, loss_all, loss_high

    def has_high_threat(self) -> bool:
        return any(self.is_high_threat.values())


class DestProbV2(POL.Policy):
    """第 2 版：两轮概率迭代 + 三家做牌推测与放铳防守（门清攻防版）。"""

    name = "dest_prob_v2"

    def __init__(self, seed: int = 0, defense_weight: float = 1.0, wall_hold_weight: float = 0.25):
        super().__init__(seed=seed)
        self.defense_weight = defense_weight
        self.wall_hold_weight = wall_hold_weight

    def _estimate_hand_value(self, env, p_idx: int, counts13: Sequence[int]) -> float:
        pl = env.players[p_idx]
        han = 1.5 if pl.is_menzen() else 0.0
        for d in _dora_tiles(env):
            han += counts13[d]
        yakuhai = {pl.seat_wind, getattr(env, "round_wind", T.EAST), T.HAKU, T.HATSU, T.CHUN}
        for y in yakuhai:
            if counts13[y] >= 3:
                han += 1.0
            elif counts13[y] == 2:
                han += 0.35
        if all(counts13[t] == 0 for t in T.TERMINALS_AND_HONORS):
            han += 1.0
        han_int = max(1, min(6, int(round(han))))
        pts = {1: 2000.0, 2: 3600.0, 3: 5800.0, 4: 8000.0, 5: 8800.0, 6: 12000.0}[han_int]
        return pts * 1.45 if pl.is_dealer else pts

    def _two_ply_dest_prob(
        self,
        counts13: List[int],
        sh0: int,
        draw_probs: Sequence[float],
        unseen: Sequence[int],
        wall_left: int,
        cache: Dict[Tuple[int, ...], int],
    ) -> Tuple[float, float]:
        turns_left = max(1.0, wall_left / 4.0)
        if sh0 == 0:
            p_win = 0.0
            for t in range(NUM_TILES):
                if unseen[t] <= 0 or counts13[t] >= 4:
                    continue
                counts13[t] += 1
                if _shanten_cached(counts13, cache) == -1:
                    p_win += draw_probs[t]
                counts13[t] -= 1
            return p_win, 1.0 - math.pow(max(0.0, 1.0 - p_win), turns_left * 1.45)

        p1_sum, p2_joint = 0.0, 0.0
        total_unseen = max(1, sum(unseen))
        do_second_ply = (sh0 <= 2)

        for t1 in range(NUM_TILES):
            p1 = draw_probs[t1]
            if p1 <= 0.0 or counts13[t1] >= 4:
                continue
            counts13[t1] += 1
            sh1 = _shanten_cached(counts13, cache)
            if sh1 < sh0:
                p1_sum += p1
                if do_second_ply:
                    best_p2 = 0.0
                    rem_t1 = max(0, unseen[t1] - 1)
                    denom_ratio = total_unseen / max(1, total_unseen - 1)
                    for d2 in range(NUM_TILES):
                        if counts13[d2] <= 0 or d2 == t1:
                            continue
                        counts13[d2] -= 1
                        if _shanten_cached(counts13, cache) == sh1:
                            p2 = 0.0
                            for t2 in range(NUM_TILES):
                                u2 = rem_t1 if t2 == t1 else unseen[t2]
                                if u2 <= 0 or counts13[t2] >= 4:
                                    continue
                                counts13[t2] += 1
                                if _shanten_cached(counts13, cache) < sh1:
                                    p_t2 = draw_probs[t2] * denom_ratio
                                    if t2 == t1 and unseen[t1] > 0:
                                        p_t2 *= (rem_t1 / unseen[t1])
                                    p2 += p_t2
                                counts13[t2] -= 1
                            if p2 > best_p2:
                                best_p2 = p2
                        counts13[d2] += 1
                    p2_joint += p1 * best_p2
            counts13[t1] -= 1

        if sh0 == 1:
            reach = 0.40 * p1_sum + 2.60 * p2_joint
            avg_p2 = (p2_joint / p1_sum) if p1_sum > 1e-9 else 0.0
            p_final = (1.0 - math.pow(max(0.0, 1.0 - p1_sum), max(1.0, turns_left * 0.65))) * (
                1.0 - math.pow(max(0.0, 1.0 - min(0.5, avg_p2)), max(1.0, turns_left * 0.70))
            )
        elif sh0 == 2:
            reach = p1_sum + 1.35 * p2_joint
            step = max(0.01, min(0.65, p1_sum))
            p_final = math.pow(1.0 - math.pow(1.0 - step, max(1.0, turns_left / 3.0)), 3)
        else:
            reach = p1_sum
            step = max(0.01, min(0.65, p1_sum))
            p_final = math.pow(1.0 - math.pow(1.0 - step, max(1.0, turns_left / (sh0 + 1))), sh0 + 1)
        return reach, p_final

    def choose(self, env, p_idx: int, legal):
        for wt in (E.ACT_TSUMO, E.ACT_RON):
            ws = [a for a in legal if a.type == wt]
            if ws:
                return ws[0]

        if env.phase == E.Phase.RESPONSE:
            return E.PASS_ACTION if E.PASS_ACTION in legal else legal[0]

        vis = _public_visible_counts(env, p_idx)
        unseen = [max(0, 4 - vis[t]) for t in range(NUM_TILES)]
        opp = OpponentInference(env, p_idx, vis, unseen)
        draw_probs = opp.adjusted_draw_probs(self.wall_hold_weight)
        wall_left = _live_wall_left(env)
        cache: Dict[Tuple[int, ...], int] = {}
        pl = env.players[p_idx]
        base_c = pl.full_counts()

        riichis = [a for a in legal if a.type == E.ACT_RIICHI]
        if riichis:
            best_r, best_ev = None, -1e18
            for a in riichis:
                c13 = list(base_c)
                c13[a.tile] -= 1
                reach, p_win = self._two_ply_dest_prob(c13, 0, draw_probs, unseen, wall_left, cache)
                val_me = self._estimate_hand_value(env, p_idx, c13)
                _, _, loss_high = opp.deal_in_risk(a.tile)
                ev = p_win * val_me + 600.0 * reach - 0.85 * self.defense_weight * loss_high
                if ev > best_ev:
                    best_ev, best_r = ev, a
            if best_r is not None:
                return best_r

        discards = [a for a in legal if a.type == E.ACT_DISCARD]
        if not discards:
            return E.PASS_ACTION if E.PASS_ACTION in legal else legal[0]
        if len(discards) == 1:
            return discards[0]

        unique_actions: Dict[int, E.Action] = {}
        for a in discards:
            if a.tile not in unique_actions:
                unique_actions[a.tile] = a

        cand_info = []
        min_sh = 99
        for t, a in unique_actions.items():
            c13 = list(base_c)
            c13[t] -= 1
            sh0 = _shanten_cached(c13, cache)
            if sh0 < min_sh:
                min_sh = sh0
            p_deal, loss_all, loss_high = opp.deal_in_risk(t)
            cand_info.append((t, a, sh0, p_deal, loss_all, loss_high))

        high_threat = opp.has_high_threat()
        eval_set = [x for x in cand_info if x[2] == min_sh]
        if high_threat and min_sh >= 1:
            safest = sorted(cand_info, key=lambda x: (x[5], x[4], x[2]))[:3]
            seen_t = {x[0] for x in eval_set}
            for x in safest:
                if x[0] not in seen_t:
                    eval_set.append(x)
                    seen_t.add(x[0])

        best_action, best_score = discards[0], -1e18
        for t, a, sh0, p_deal, loss_all, loss_high in eval_set:
            c13 = list(base_c)
            c13[t] -= 1
            if sh0 > min_sh + 1:
                reach, p_win = 0.0, 0.002
            else:
                reach, p_win = self._two_ply_dest_prob(c13, sh0, draw_probs, unseen, wall_left, cache)
            val_me = self._estimate_hand_value(env, p_idx, c13)
            defense_cost = (
                self.defense_weight * (1.25 * loss_high + 0.15 * (loss_all - loss_high))
                if high_threat
                else self.defense_weight * 0.22 * loss_all
            )
            sh_penalty = (sh0 - min_sh) * (450.0 if (high_threat and min_sh >= 2) else 1100.0)
            total_ev = p_win * val_me + 2200.0 * reach - sh_penalty - defense_cost - (110.0 if t in opp.doras else 0.0)
            if total_ev > best_score:
                best_score, best_action = total_ev, a
        return best_action


def register_policy(name: str = "dest_prob_v2") -> None:
    POL.POLICIES[name] = DestProbV2


if __name__ == "__main__":
    import arena

    register_policy("dest_prob_v2")
    res = arena.evaluate(["dest_prob_v2", "eff_wuke", "eff_uke", "greedy_tenpai"], n_batches=10, seed0=100)
    print(res.report())
