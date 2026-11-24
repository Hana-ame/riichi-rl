"""观测编码器：把局面输出成 (C, 34) 平面张量 + 标量向量。

设计依据（2026-10-04 调研）：
- LLM_Mahjong 的 bc49 用 **56 平面 × 34 张 + 29 标量**（二维，卷积吃得到结构）
- 我原来的 436 维拍平向量**不能直接卷积**（436 不被 34 整除），
  所以这里重写成真正的二维平面编码。

铁律：每个平面都必须是**公开或自有的信息**。
禁止：对手手牌、牌山顺序、未来牌、任何牌效统计量。

修复记录：
- 宝牌平面使用 mahjong.tiles.dora_from_indicator（修正 9m->1m、北->东、中->白 循环）
  并覆盖所有已翻出的杠宝牌指示牌；
- 四家手牌张数、牌河、副露、点数与弃牌顺序统一按相对视角旋转
  （0=自家, 1=下家, 2=对家, 3=上家），避免绝对座位编号导致网络难以对齐敌我。
"""

from __future__ import annotations

import numpy as np

from mahjong import env as E
from mahjong import ruleset as RS
from mahjong import tiles as T
from mahjong.tiles import NUM_TILES

# 22 基础平面 + 3 对手立直平面(22..24) + 3 对手现物安全牌平面(25..27) + 1 赤宝牌平面(28)
N_PLANES = 29
PLANE = NUM_TILES
# 26 基础标量 + 3 对手立直标志(26..28) + 3 响应弃牌者相对座位(29..31)
SCALAR_DIM = 32


def _tile_plane():
    return np.eye(NUM_TILES, dtype=np.float32)


ORDER_PLANES = 4 * 6   # v3：四家各最近 6 张弃牌位次 one-hot


class ObsEncoder:
    """局面 -> (N_PLANES, 34) float32 + (SCALAR_DIM,) float32

    order=True 时追加 24 个顺序平面（obs v3，共 46 平面）。
    """

    def __init__(self, order=False):
        self.eye = _tile_plane()
        self.order = bool(order)
        self.n_planes = N_PLANES + (ORDER_PLANES if order else 0)

    def encode(self, env, p_idx):
        p = env.players[p_idx]
        # 相对座位顺序：0=自家, 1=下家, 2=对家, 3=上家
        rel_players = [env.players[(p_idx + offset) % 4] for offset in range(4)]
        planes = np.zeros((self.n_planes, NUM_TILES), dtype=np.float32)

        # 0: 自己手牌计数（0..4）
        for t, c in enumerate(p.counts()):
            planes[0, t] = c

        # 1-4: 四家暗牌张数（相对视角）
        for i, q in enumerate(rel_players):
            planes[1 + i, :] = len(q.hand) / 14.0

        # 5-8: 四家牌河计数（相对视角）
        for i, q in enumerate(rel_players):
            for t in q.river:
                planes[5 + i, t] += 1.0

        # 9-12: 四家副露计数（相对视角）
        for i, q in enumerate(rel_players):
            for kind, t, _ in q.melds:
                n = 4 if kind in ("ankan", "minkan", "addkan") else 3
                if kind == "chi":
                    planes[9 + i, t] += 1
                    planes[9 + i, t + 1] += 1
                    planes[9 + i, t + 2] += 1
                else:
                    planes[9 + i, t] += n

        # 13: 各家副露组数（相对视角放在 0..3 号位）
        for i, q in enumerate(rel_players):
            planes[13, i] = len(q.melds) / 4.0

        # 14: 场上已见牌总数（弃牌/副露合计，按牌种）
        seen = np.zeros(NUM_TILES, dtype=np.float32)
        for q in rel_players:
            for t in q.river:
                seen[t] += 1
            for kind, t, _ in q.melds:
                if kind == "chi":
                    seen[t] += 1; seen[t + 1] += 1; seen[t + 2] += 1
                else:
                    seen[t] += 4 if kind in ("ankan", "minkan", "addkan") else 3
        planes[14] = seen

        # 15: 宝牌指示牌（包含全部已翻开的杠宝牌指示牌）
        dora_inds = getattr(env, "dora_indicators", None)
        if dora_inds:
            for ind in dora_inds:
                if ind is not None and 0 <= ind < NUM_TILES:
                    planes[15, ind] += 1.0
        elif env.dora_indicator is not None:
            planes[15, env.dora_indicator] = 1.0

        # 16: 里宝牌指示牌 one-hot（仅公开可见时非 None）
        if env.ura_dora_indicator is not None:
            planes[16, env.ura_dora_indicator] = 1.0

        # 17: 上一张弃牌 one-hot
        if env.last_discard is not None:
            planes[17, env.last_discard] = 1.0

        # 18: 自家牌河计数
        for t in p.river:
            planes[18, t] += 1.0

        # 19: 是否立直（每张牌一个标志，便于注意力聚焦）
        if p.riichi:
            planes[19, :] = 1.0

        # 20: 宝牌集合（按正确规则循环映射，覆盖所有已翻指示牌）
        if dora_inds:
            for ind in dora_inds:
                if ind is not None and 0 <= ind < NUM_TILES:
                    planes[20, T.dora_from_indicator(ind)] += 1.0
        elif env.dora_indicator is not None:
            planes[20, T.dora_from_indicator(env.dora_indicator)] = 1.0
        if env.ura_dora_indicator is not None:
            planes[20, T.dora_from_indicator(env.ura_dora_indicator)] += 1.0

        # 21: 剩余未知牌数（不可见牌 = 4 - 已见 - 自己手里）
        planes[21] = np.clip(4.0 - seen - np.array(
            p.counts(), dtype=np.float32), 0, 4)

        # 22-24: 三家对手立直标志（相对视角：下家/对家/上家）
        for opp_i in range(1, 4):
            q = rel_players[opp_i]
            if q.riichi:
                planes[21 + opp_i, :] = 1.0

        # 25-27: 三家对手现物安全牌平面（对手牌河 + 该对手立直后全员弃牌）
        for opp_i in range(1, 4):
            q = rel_players[opp_i]
            safe_tiles = set(q.river)
            gb = getattr(q, "genbutsu", None)
            if gb:
                safe_tiles.update(gb)
            for t in safe_tiles:
                if 0 <= t < NUM_TILES:
                    planes[24 + opp_i, t] = 1.0

        # 28: 自家手牌中的赤宝牌候选（5m/5p/5s）
        for t in RS.AKA_TILES:
            if t in p.hand:
                planes[28, t] = 1.0

        # --- 标量 ---
        s = np.zeros(SCALAR_DIM, dtype=np.float32)
        s[0:4] = [q.points / 25000.0 for q in rel_players]
        s[4] = 1.0 if p.is_dealer else 0.0
        s[5] = min(1.0, env.turn / 72.0)
        s[6:10] = [1.0 if env.round_wind == t else 0.0
                   for t in (27, 28, 29, 30)]
        s[10:14] = [1.0 if p.seat_wind == t else 0.0
                    for t in (27, 28, 29, 30)]
        s[14] = 1.0 if p.furiten else 0.0
        s[15] = 1.0 if p.riichi else 0.0
        if hasattr(env, "_drawable_left"):
            wall_left = env._drawable_left()
        else:
            dead = getattr(env, "dead_wall_size", 0)
            wall_left = max(0, len(env.wall) - dead - env.wall_idx)
        s[16] = min(1.0, wall_left / 70.0)
        s[17:21] = [len(q.hand) / 14.0 for q in rel_players]
        s[21] = 1.0 if env.phase == E.Phase.DISCARD else 0.0
        s[22] = 1.0 if env.phase == E.Phase.RESPONSE else 0.0
        s[23] = len(p.melds) / 4.0
        s[24] = 1.0 if env.last_discard is not None else 0.0
        s[25] = 1.0 if env.last_discard_player == p_idx else 0.0
        # 26-28: 三家对手立直标志（下家/对家/上家）
        for opp_i in range(1, 4):
            s[25 + opp_i] = 1.0 if rel_players[opp_i].riichi else 0.0
        # 29-31: 响应阶段弃牌者相对座位 one-hot（下家/对家/上家）
        if env.phase == E.Phase.RESPONSE and env.last_discard_player is not None:
            rel_disc = (env.last_discard_player - p_idx) % 4
            if 1 <= rel_disc <= 3:
                s[28 + rel_disc] = 1.0

        if self.order:
            for i, q in enumerate(rel_players):
                seq = q.river[-6:]
                for pos, t in enumerate(seq):
                    planes[N_PLANES + i * 6 + pos, t] = 1.0
        return planes, s


OBS_TENSOR_DIM = N_PLANES * PLANE + SCALAR_DIM
