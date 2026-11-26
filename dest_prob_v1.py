"""非 ML 日麻策略 · 第 1 版（v1）：胡牌目标（dest = Win）+ 两轮概率迭代（2-Ply Lookahead）。

相比第 0 版的升级（v1 / 共 4 版）：
- 新增**两轮迭代前瞻**：
  * 第 1 轮：枚举切牌 d 后所有降向听有效进张 t1 及摸牌概率 P(draw t1)。
  * 第 2 轮：在摸进 t1 后的 14 张手牌中，枚举保持新向听 sh1 的最优续切 d2，
    再累加第 2 轮降向听（sh0 == 1 时即直接到达 dest = 胡牌）的条件概率 P(draw t2 | t1)。
- 新增手牌宝牌保留微调（避免同牌效时误切宝牌）。
- 仍不含三家做牌推测与放铳防守（作为纯两轮进攻对照版）。
"""

from __future__ import annotations

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


def _shanten_cached(counts: Sequence[int], cache: Dict[Tuple[int, ...], int]) -> int:
    key = tuple(counts)
    val = cache.get(key)
    if val is None:
        val = REF.ref_shanten(counts)
        cache[key] = val
    return val


class DestProbV1(POL.Policy):
    """第 1 版：以胡牌为 dest 的两轮摸牌概率迭代（2-Ply 纯进攻版）。"""

    name = "dest_prob_v1"

    def _two_ply_score(
        self,
        counts13: List[int],
        sh0: int,
        draw_probs: Sequence[float],
        unseen: Sequence[int],
        cache: Dict[Tuple[int, ...], int],
    ) -> float:
        """两轮迭代计算 13 张手牌向 dest=胡牌（shanten = -1）推进的期望概率得分。"""
        # 听牌（sh0 == 0）：直接计算 1 摸胡牌概率
        if sh0 == 0:
            p_win = 0.0
            for t in range(NUM_TILES):
                if unseen[t] <= 0 or counts13[t] >= 4:
                    continue
                counts13[t] += 1
                if _shanten_cached(counts13, cache) == -1:
                    p_win += draw_probs[t]
                counts13[t] -= 1
            return p_win

        p1_sum = 0.0
        p2_joint = 0.0
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
            # 1 向听时，p2_joint 即「两步直达胡牌」联合概率
            return 0.40 * p1_sum + 2.60 * p2_joint
        if sh0 == 2:
            return p1_sum + 1.35 * p2_joint
        return p1_sum

    def choose(self, env, p_idx: int, legal):
        for wt in (E.ACT_TSUMO, E.ACT_RON):
            ws = [a for a in legal if a.type == wt]
            if ws:
                return ws[0]

        if env.phase == E.Phase.RESPONSE:
            return E.PASS_ACTION if E.PASS_ACTION in legal else legal[0]

        vis = _public_visible_counts(env, p_idx)
        unseen = [max(0, 4 - vis[t]) for t in range(NUM_TILES)]
        total_unseen = max(1, sum(unseen))
        draw_probs = [u / total_unseen for u in unseen]
        doras = {T.dora_from_indicator(d) for d in getattr(env, "dora_indicators", ())}
        cache: Dict[Tuple[int, ...], int] = {}

        pl = env.players[p_idx]
        base_c = pl.full_counts()

        # 立直：选待牌胡牌概率最高者
        riichis = [a for a in legal if a.type == E.ACT_RIICHI]
        if riichis:
            best_r, best_p = None, -1.0
            for a in riichis:
                c13 = list(base_c)
                c13[a.tile] -= 1
                p_win = self._two_ply_score(c13, 0, draw_probs, unseen, cache)
                if p_win > best_p:
                    best_p, best_r = p_win, a
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

        # 只对最小向听数的候选做两轮迭代比较
        cand_sh = []
        min_sh = 99
        for t, a in unique_actions.items():
            c13 = list(base_c)
            c13[t] -= 1
            sh0 = _shanten_cached(c13, cache)
            if sh0 < min_sh:
                min_sh = sh0
            cand_sh.append((t, a, sh0))

        best_action = discards[0]
        best_val = -1e18
        for t, a, sh0 in cand_sh:
            if sh0 != min_sh:
                continue
            c13 = list(base_c)
            c13[t] -= 1
            score = self._two_ply_score(c13, sh0, draw_probs, unseen, cache)
            # 同牌效时轻微偏好保留宝牌
            if t in doras:
                score -= 0.015
            if score > best_val:
                best_val = score
                best_action = a
        return best_action


def register_policy(name: str = "dest_prob_v1") -> None:
    POL.POLICIES[name] = DestProbV1


if __name__ == "__main__":
    import arena

    register_policy("dest_prob_v1")
    res = arena.evaluate(["dest_prob_v1", "eff_wuke", "eff_uke", "greedy_tenpai"], n_batches=10, seed0=100)
    print(res.report())
