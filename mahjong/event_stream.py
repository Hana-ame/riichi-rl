"""统一历史事件流编码器：把牌河、手切/摸切、立直与副露动作压成固定长度离散序列。

输出张量规格：
    events: int64, shape (T, 4)，默认 T = 48（可配 48..64）
    每行 4 个离散属性：
        0: rel_seat   (0=PAD/BOS, 1=自家, 2=下家, 3=对家, 4=上家)
        1: tile_id    (0..33 牌种, 34=无牌/BOS)
        2: act_type   (0=BOS/PAD, 1=手切discard, 2=摸切tsumogiri, 3=立直, 4=吃, 5=碰, 6=杠)
        3: turn_idx   (0=PAD, 1..24 巡目桶)

设计要点：
- 序列第 0 位固定放置 BOS 起手标记 (rel_seat=1, tile=34, act=0, turn=1)，
  确保开局第 0 巡尚未有任何弃牌时，Cross-Attention 的 key_padding_mask 也不会全 True 导致 NaN。
- 严格仅使用公开可见信息（牌河、副露、立直状态），绝不泄露对手手牌或牌山顺序。
"""

from __future__ import annotations

import numpy as np

from mahjong import tiles as T

DEFAULT_EVENT_LEN = 48

# 词表大小（供 nn.Embedding 使用）
SEAT_VOCAB = 5     # 0=PAD, 1..4=相对四家
TILE_VOCAB = 35    # 0..33 牌种, 34=BOS/空
ACT_VOCAB = 7      # 0=BOS/PAD, 1=discard, 2=tsumogiri, 3=riichi, 4=chi, 5=pon, 6=kan
TURN_VOCAB = 25    # 0=PAD, 1..24 巡目

_TILE_NAME_TO_ID = {name: i for i, name in enumerate(T.TILE_NAMES)}


def _parse_tile_id(val) -> int:
    if isinstance(val, (int, np.integer)):
        return int(val) if 0 <= int(val) < 34 else 34
    if isinstance(val, str):
        return _TILE_NAME_TO_ID.get(val, 34)
    return 34


def encode_event_stream(env, p_idx: int, max_len: int = DEFAULT_EVENT_LEN) -> np.ndarray:
    """从 MahjongEnv 提取视角为 p_idx 的最近 max_len 步公开事件流 -> int64 (max_len, 4)。"""
    out = np.zeros((max_len, 4), dtype=np.int64)
    # 槽位 0：BOS 起始标记，防止开局空历史导致注意力掩码全空
    out[0] = (1, 34, 0, 1)

    events = []
    raw_log = getattr(env, "log", None) or []
    if raw_log:
        for entry in raw_log:
            if not isinstance(entry, (tuple, list)) or len(entry) < 4:
                continue
            turn, actor, kind, tile_raw = entry[0], entry[1], entry[2], entry[3]
            rel_seat = ((int(actor) - int(p_idx)) % 4) + 1
            tile_id = _parse_tile_id(tile_raw)
            if kind == "tsumogiri":
                act_id = 2
            elif kind == "riichi":
                act_id = 3
            elif kind == "chi":
                act_id = 4
            elif kind == "pon":
                act_id = 5
            elif kind in ("kan", "ankan", "minkan", "addkan"):
                act_id = 6
            else:
                act_id = 1
            turn_b = max(1, min(24, int(turn) + 1))
            events.append((rel_seat, tile_id, act_id, turn_b))
    else:
        # 兼容无 log 的测试构造环境：按四家牌河与副露合成历史
        for offset in range(4):
            q_idx = (p_idx + offset) % 4
            q = env.players[q_idx]
            rel_seat = offset + 1
            for t_idx, tile_id in enumerate(q.river):
                act_id = 3 if getattr(q, "riichi_turn", -1) == t_idx else 1
                turn_b = max(1, min(24, t_idx + 1))
                events.append((rel_seat, int(tile_id), act_id, turn_b))
            for mkind, mtile, _ in q.melds:
                act_id = 4 if mkind == "chi" else (5 if mkind == "pon" else 6)
                events.append((rel_seat, int(mtile), act_id, 1))

    # 截取最近 max_len - 1 条事件紧跟在 BOS 之后
    keep = events[-(max_len - 1):] if max_len > 1 else []
    for idx, ev in enumerate(keep, start=1):
        out[idx] = ev
    return out
