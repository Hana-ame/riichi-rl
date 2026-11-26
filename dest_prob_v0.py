"""非 ML 日麻策略 · 第 0 版（v0）：胡牌目标（dest = Win）+ 1 轮摸牌概率基线。

版本定位（v0 / 共 4 版）：
- 仅使用「严格公开可见牌」计算 34 种牌的未知剩余枚数 U[t] 与均匀摸牌概率 P(draw t) = U[t] / sum(U)。
- 每步最小化离胡牌（dest = Win, shanten = -1）的向听数；同向听时最大化「1 轮摸牌直接降向听的概率」。
- 不做第 2 轮迭代、不做三家做牌推测与防守、不主动吃碰（保持门清立直）。

可直接接入 `riichi-rl` 的 `MahjongEnv` 与 `arena.py`：
    from dest_prob_v0 import DestProbV0, register_policy
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from mahjong import env as E
from mahjong import policies as POL
from mahjong import refengine as REF
from mahjong.tiles import NUM_TILES


def _public_visible_counts(env, p_idx: int) -> List[int]:
    """严格公开可见牌计数（自家暗牌 + 全员牌河 + 全员副露 + 宝牌指示牌）。"""
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


class DestProbV0(POL.Policy):
    """第 0 版：以胡牌为 dest 的向听最小化 + 1 轮摸牌概率最大化（纯进攻基线）。"""

    name = "dest_prob_v0"

    def _one_ply_prob(
        self,
        counts13: List[int],
        sh0: int,
        draw_probs: Sequence[float],
        cache: Dict[Tuple[int, ...], int],
    ) -> float:
        """计算 13 张手牌在下一摸（1 轮）严格向 dest=胡牌（shanten = -1）推进的概率。"""
        p_improve = 0.0
        for t in range(NUM_TILES):
            p = draw_probs[t]
            if p <= 0.0 or counts13[t] >= 4:
                continue
            counts13[t] += 1
            if _shanten_cached(counts13, cache) < sh0:
                p_improve += p
            counts13[t] -= 1
        return p_improve

    def choose(self, env, p_idx: int, legal):
        # 1) 必和：自摸 / 荣和
        for wt in (E.ACT_TSUMO, E.ACT_RON):
            ws = [a for a in legal if a.type == wt]
            if ws:
                return ws[0]

        # 响应阶段：除荣和外一律 pass（保持门清）
        if env.phase == E.Phase.RESPONSE:
            return E.PASS_ACTION if E.PASS_ACTION in legal else legal[0]

        vis = _public_visible_counts(env, p_idx)
        unseen = [max(0, 4 - vis[t]) for t in range(NUM_TILES)]
        total_unseen = max(1, sum(unseen))
        draw_probs = [u / total_unseen for u in unseen]
        cache: Dict[Tuple[int, ...], int] = {}
        pl = env.players[p_idx]
        base_c = pl.full_counts()

        # 2) 能立直就选「1 轮胡牌概率最高」的立直切牌
        riichis = [a for a in legal if a.type == E.ACT_RIICHI]
        if riichis:
            best_r, best_p = None, -1.0
            for a in riichis:
                c13 = list(base_c)
                c13[a.tile] -= 1
                p_win = self._one_ply_prob(c13, 0, draw_probs, cache)
                if p_win > best_p:
                    best_p, best_r = p_win, a
            if best_r is not None:
                return best_r

        # 3) 切牌：最小向听 -> 最大 1 轮推进概率
        discards = [a for a in legal if a.type == E.ACT_DISCARD]
        if not discards:
            return E.PASS_ACTION if E.PASS_ACTION in legal else legal[0]
        if len(discards) == 1:
            return discards[0]

        unique_actions: Dict[int, E.Action] = {}
        for a in discards:
            if a.tile not in unique_actions:
                unique_actions[a.tile] = a

        best_action = discards[0]
        best_key = None
        for t, a in unique_actions.items():
            c13 = list(base_c)
            c13[t] -= 1
            sh0 = _shanten_cached(c13, cache)
            p1 = self._one_ply_prob(c13, sh0, draw_probs, cache)
            key = (sh0, -p1)
            if best_key is None or key < best_key:
                best_key = key
                best_action = a
        return best_action


def register_policy(name: str = "dest_prob_v0") -> None:
    POL.POLICIES[name] = DestProbV0


if __name__ == "__main__":
    import arena

    register_policy("dest_prob_v0")
    res = arena.evaluate(["dest_prob_v0", "eff_wuke", "eff_uke", "greedy_tenpai"], n_batches=10, seed0=100)
    print(res.report())
