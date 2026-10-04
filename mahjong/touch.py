"""手牌形状特征（touch features）。

为什么需要它
----------
纯 RL 只从「输赢」学，需要极多局才能摸到手牌结构。
（参考：LLM_Mahjong 的纯血线跑 32-64M 局才到 share 0.4568）

这些特征**只回答「我的手牌长什么样」**，不含任何「该打哪张」：
- 已有面子数（刻子/顺子/杠）
- 搭子数（两面/嵌张/边张的对子外）
- 听牌待牌种数
- 向听数

每一维都能对任意手牌算出来，不需要人类知识表，
因此不违反「不许用牌效表」——牌效表是「牌 A 与牌 B 的相关性」，
这里是「我这手牌已经凑成了几组」。

维度：TOUCH_DIM = 8
"""

from __future__ import annotations

TOUCH_DIM = 8


def shape_features(counts, meld_count, shanten, waits):
    """返回 TOUCH_DIM 维 float32。

    counts: 34 维计数（含副露牌）
    meld_count: 副露组数
    shanten: 向听数（-1 已和、0 听牌、8 未知）
    waits: 听牌待牌种数（0 表示未听）
    """
    import numpy as np

    # 已有刻子/杠
    triplets = sum(1 for t in range(34) if counts[t] >= 3)
    # 已有对子（>=2 张的牌种数）
    pairs = sum(1 for t in range(34) if counts[t] >= 2)
    # 单张（孤张）
    singles = sum(1 for t in range(34) if counts[t] == 1)
    # 顺子数：每门里能数出的最多不重叠顺子
    seqs = 0
    for suit in range(3):
        for r in range(7):
            a, b, c = suit * 9 + r, suit * 9 + r + 1, suit * 9 + r + 2
            k = min(counts[a], counts[b], counts[c])
            seqs += k

    f = [
        float(meld_count) / 4.0,
        float(triplets) / 4.0,
        float(seqs) / 8.0,
        float(pairs) / 7.0,
        float(singles) / 14.0,
        1.0 if shanten == 0 else 0.0,
        min(1.0, waits / 8.0),
        min(1.0, max(0.0, (8 - shanten) / 8.0)) if shanten > 0 else 1.0,
    ]
    return np.asarray(f, dtype=np.float32)
