"""向听数（shanten）与相关工具。

与规则层（hand.py / scoring.py）的区别：
- hand.py 回答「这手牌和了吗」（布尔）
- 本模块回答「这手牌离听牌还差几步」（整数）

2026-11-27 修复：原手写公式遗漏搭子（taatsu）并重复扣减面子数，会把 1 向听
误算成 0 向听。现统一委托给仓库内已验证的 MahjongRepository/mahjong 2.0.0
实现（mahjong/shanten_ref.py），并保持原接口签名兼容。
"""

from __future__ import annotations

from .shanten_ref import Shanten
from .tiles import NUM_TILES, TERMINALS_AND_HONORS, is_honor  # noqa: F401

INF = 8


def _normalize_counts(counts, melds=0):
    c = [int(x) for x in counts]
    total = sum(c)
    if total == 0:
        return None
    # shanten_ref 按 (14 - total) // 3 推断已副露组数，要求 total % 3 != 0 且 <= 14
    max_concealed = max(1, 14 - 3 * int(melds))
    if total > max_concealed:
        # 兼容极少数传入含副露的完整计数的调用
        total_ok = total <= 14 and total % 3 != 0
        if not total_ok:
            return None
    if total > 14 or total % 3 == 0:
        return None
    return c


def standard_shanten(counts, melds=0, meld_known=None):
    """标准型向听数（-1 = 已和了，0 = 听牌，>=1 = 向听数）。"""
    c = _normalize_counts(counts, melds=melds)
    if c is None:
        return INF
    return Shanten.calculate_shanten_for_regular_hand(c)


def shanten(counts, melds=0):
    """总向听数 = 标准型 / 七对 / 国士 三者取最小（-1 = 已和了，0 = 听牌）。"""
    c = _normalize_counts(counts, melds=melds)
    if c is None:
        return INF
    use_special = (int(melds) == 0 and sum(c) >= 13)
    return Shanten.calculate_shanten(
        c, use_chiitoitsu=use_special, use_kokushi=use_special
    )


def _chiitoi_shanten(counts, melds=0):
    if melds > 0:
        return INF
    return Shanten.calculate_shanten_for_chiitoitsu_hand(counts)


def _kokushi_shanten(counts):
    return Shanten.calculate_shanten_for_kokushi_hand(counts)
