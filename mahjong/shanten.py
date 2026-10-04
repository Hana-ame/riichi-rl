"""向听数（shanten）与相关工具。

与规则层（hand.py / scoring.py）的区别：
- hand.py 回答「这手牌和了吗」（布尔）
- 本模块回答「这手牌离听牌还差几步」（整数）

向听数是**规则层的定义**（同花顺/七对/国士各自的计算方式），
不是策略先验。用它做奖励整形是训练设计，不是麻将知识。
"""

from __future__ import annotations

from .tiles import NUM_TILES, TERMINALS_AND_HONORS, is_honor

INF = 8


def _seq_starts():
    return [suit * 9 + r for suit in range(3) for r in range(7)]


_SEQ = tuple(_seq_starts())


def standard_shanten(counts, melds=0, meld_known=None):
    """标准型向听数。

    counts: 34 维计数（含雀头候选与暗牌）
    melds: 已成组数（副露 + 暗刻暗杠），用于从总组数里扣掉

    标准公式：
        shanten = (4 - 面子数) * 2 - 雀头数 - min(面子数, 雀头数) + 约束修正
    这里用「最大可成面子数 + 最大可成雀头数」直接算，避免公式的边界错误。
    """
    c = list(counts)
    total = sum(c)
    if total == 0:
        return INF

    best = INF
    # 枚举雀头
    for pt in range(NUM_TILES):
        if c[pt] < 2:
            continue
        c[pt] -= 2
        m = _max_melds(c)
        v = 8 - (m + melds) * 2 - (m if m < 4 else 4)
        # 约束修正：有面子就不需要雀头；无雀头时少 1 分
        if m >= 4:
            v = 8 - 8 - 1
        if v < best:
            best = v
        c[pt] += 2
    # 无雀头
    m = _max_melds(c)
    v = 8 - (m + melds) * 2
    if v < best:
        best = v
    return max(0, best) if best > 0 else 0


def _max_melds(c):
    """从计数向量里最多能取出多少组面子（顺/刻/杠）。"""
    c = list(c)
    best = 0
    def rec(idx, start, used):
        nonlocal best
        if used > best:
            best = used
        if idx >= NUM_TILES or used >= 5:
            return
        # 跳过空牌
        while idx < NUM_TILES and c[idx] == 0:
            idx += 1
        if idx >= NUM_TILES:
            return
        t = idx
        # 刻子
        if c[t] >= 3:
            c[t] -= 3
            rec(idx, t, used + 1)
            c[t] += 3
        # 顺子
        if t in _SEQ and c[t + 1] > 0 and c[t + 2] > 0:
            c[t] -= 1; c[t + 1] -= 1; c[t + 2] -= 1
            rec(idx, t, used + 1)
            c[t] += 1; c[t + 1] += 1; c[t + 2] += 1
        # 只用当前最小牌（不跳过），避免漏解
        rec(t + 1, t, used)
    rec(0, 0, 0)
    return best


def shanten(counts, melds=0):
    """总向听数 = 标准型 / 七对 / 国士 三者取最小。"""
    a = standard_shanten(counts, melds)
    b = _chiitoi_shanten(counts, melds)
    c = _kokushi_shanten(counts)
    return min(a, b, c)


def _chiitoi_shanten(counts, melds=0):
    if melds > 0:
        return INF
    pairs = 0
    kinds = 0
    for i in range(NUM_TILES):
        if counts[i] == 0:
            continue
        kinds += 1
        if counts[i] >= 2:
            pairs += 1
    return 6 - pairs + max(0, 7 - kinds)


def _kokushi_shanten(counts):
    kinds = 0
    has_pair = 0
    for t in TERMINALS_AND_HONORS:
        if counts[t] > 0:
            kinds += 1
            if counts[t] >= 2:
                has_pair = 1
    return 13 - kinds - has_pair
