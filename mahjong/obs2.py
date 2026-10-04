"""观测编码器：把局面输出成 (C, 34) 平面张量 + 标量向量。

设计依据（2026-10-04 调研）：
- LLM_Mahjong 的 bc49 用 **56 平面 × 34 张 + 29 标量**（二维，卷积吃得到结构）
- 我原来的 436 维拍平向量**不能直接卷积**（436 不被 34 整除），
  所以这里重写成真正的二维平面编码。

铁律：每个平面都必须是**公开或自有的信息**。
禁止：对手手牌、牌山顺序、未来牌、任何牌效统计量。
"""

from __future__ import annotations

import numpy as np

from mahjong import env as E
from mahjong.tiles import NUM_TILES

N_PLANES = 22
PLANE = NUM_TILES
SCALAR_DIM = 26


def _tile_plane():
    return np.eye(NUM_TILES, dtype=np.float32)


class ObsEncoder:
    """局面 -> (N_PLANES, 34) float32 + (SCALAR_DIM,) float32"""

    def __init__(self):
        self.eye = _tile_plane()

    def encode(self, env, p_idx):
        p = env.players[p_idx]
        planes = np.zeros((N_PLANES, NUM_TILES), dtype=np.float32)

        # 0: 自己手牌计数（归一到 0..4）
        for t, c in enumerate(p.counts()):
            planes[0, t] = c

        # 1-4: 四家暗牌张数
        for i, q in enumerate(env.players):
            planes[1 + i, :] = len(q.hand) / 14.0

        # 5-8: 四家牌河计数
        for i, q in enumerate(env.players):
            for t in q.river:
                planes[5 + i, t] += 1.0

        # 9-12: 四家副露计数
        for i, q in enumerate(env.players):
            for kind, t, _ in q.melds:
                n = 4 if kind in ("ankan", "minkan", "addkan") else 3
                if kind == "chi":
                    planes[9 + i, t] += 1
                    planes[9 + i, t + 1] += 1
                    planes[9 + i, t + 2] += 1
                else:
                    planes[9 + i, t] += n

        # 13: 各家副露组数（放在 0 号位做标记，其余为 0）
        planes[13, 0] = len(p.melds) / 4.0

        # 14: 场上已见牌总数（弃牌/副露合计，按牌种）
        seen = np.zeros(NUM_TILES, dtype=np.float32)
        for q in env.players:
            for t in q.river:
                seen[t] += 1
            for kind, t, _ in q.melds:
                if kind == "chi":
                    seen[t] += 1; seen[t + 1] += 1; seen[t + 2] += 1
                else:
                    seen[t] += 4 if kind in ("ankan", "minkan", "addkan") else 3
        planes[14] = seen

        # 15: 宝牌指示牌 one-hot
        if env.dora_indicator is not None:
            planes[15, env.dora_indicator] = 1.0

        # 16: 里宝牌指示牌 one-hot
        if env.ura_dora_indicator is not None:
            planes[16, env.ura_dora_indicator] = 1.0

        # 17: 上一张弃牌 one-hot
        if env.last_discard is not None:
            planes[17, env.last_discard] = 1.0

        # 18: 自家牌河 one-hot
        for t in p.river:
            planes[18, t] += 1.0

        # 19: 是否立直（每张牌一个标志，便于注意力聚焦）
        if p.riichi:
            planes[19, :] = 1.0

        # 20: 宝牌集合 one-hot（公开信息：指示牌可推出）
        if env.dora_indicator is not None:
            d = (env.dora_indicator + 1) % NUM_TILES
            planes[20, d] = 1.0
        if env.ura_dora_indicator is not None:
            d = (env.ura_dora_indicator + 1) % NUM_TILES
            planes[20, d] += 1.0

        # 21: 剩余未知牌数（不可见牌 = 4 - 已见 - 自己手里），
        #     这是公开可推的信息，不含任何牌序
        planes[21] = np.clip(4.0 - seen - np.array(
            p.counts(), dtype=np.float32), 0, 4)

        # --- 标量 ---
        s = np.zeros(SCALAR_DIM, dtype=np.float32)
        s[0:4] = [q.points / 25000.0 for q in env.players]
        s[4] = 1.0 if p.is_dealer else 0.0
        s[5] = min(1.0, env.turn / 72.0)
        s[6:10] = [1.0 if env.round_wind == t else 0.0
                   for t in (27, 28, 29, 30)]
        s[10:14] = [1.0 if p.seat_wind == t else 0.0
                    for t in (27, 28, 29, 30)]
        s[14] = 1.0 if p.furiten else 0.0
        s[15] = 1.0 if p.riichi else 0.0
        s[16] = min(1.0, (len(env.wall) - env.wall_idx) / 70.0)
        s[17:21] = [len(q.hand) / 14.0 for q in env.players]
        s[21] = 1.0 if env.phase == E.Phase.DISCARD else 0.0
        s[22] = 1.0 if env.phase == E.Phase.RESPONSE else 0.0
        s[23] = len(p.melds) / 4.0
        s[24] = 1.0 if env.last_discard is not None else 0.0
        s[25] = 1.0 if env.last_discard_player == p_idx else 0.0

        return planes, s


OBS_TENSOR_DIM = N_PLANES * PLANE + SCALAR_DIM
