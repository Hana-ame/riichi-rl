"""手牌判定：和了识别、全部和了分解枚举、向听数（shanten）。

对应规则文档：riichi-rl/rules/RULES.md §6、§11

设计要点
--------
1. 和了分解 = 一组面子（顺/刻/杠）+ 一组雀头。副露组在 env 层单独处理，
   这里只处理「当前这一组」是否可分解，避免门前/副露重复计数。
2. 枚举器必须**穷尽**，因为算番要在所有分解里取最高分。
3. 向听数用于「立直可否」「宣言流局可否」等**合法性**判断，
   这是规则判定，不是策略先验（RULES.md §15）。
"""

from __future__ import annotations

from .tiles import NUM_TILES, TERMINALS_AND_HONORS, is_honor

CHI = "chi"
PONG = "pon"
KAN = "kan"
PAIR = "pair"

# 顺子的起始牌 id：每门花色的 1..7，共 24 个。
# 注意是「整数集合」——分解时用 t in _SEQ_STARTS 判断 t 能否起顺。
_SEQ_STARTS = frozenset(suit * 9 + r for suit in range(3) for r in range(7))

#: 国士所需的 13 种幺九牌，作为集合便于 O(1) 判定
_Y9_SET = frozenset((0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33))


# ---------------------------------------------------------------------------
# 全量枚举（回溯）
# ---------------------------------------------------------------------------

def enumerate_standard(counts, meld_count):
    """穷举一组 meld 能补满的分解。

    参数:
        counts: 34 维暗牌计数（尚未取副露，暗牌含 雀头候选）
        meld_count: 需要凑的组数（副露组数 + 本组面子数）

    返回: list of Decomposition
    """
    results = []
    acc = []
    _recurse(counts, 0, meld_count, acc, results)
    return results


def _recurse(c, start, need, acc, out):
    if need == 0:
        if sum(c) == 0:
            out.append(Decomposition(tuple(acc)))
        return
    t = -1
    for i in range(start, NUM_TILES):
        if c[i] > 0:
            t = i
            break
    if t < 0:
        return
    # 刻子
    if c[t] >= 3:
        c[t] -= 3
        acc.append((PONG, t))
        _recurse(c, t, need - 1, acc, out)
        acc.pop()
        c[t] += 3
    # 顺子
    if t in _SEQ_STARTS and c[t + 1] > 0 and c[t + 2] > 0:
        c[t] -= 1; c[t + 1] -= 1; c[t + 2] -= 1
        acc.append((CHI, t))
        _recurse(c, t, need - 1, acc, out)
        acc.pop()
        c[t] += 1; c[t + 1] += 1; c[t + 2] += 1


class Decomposition:
    """一组面子的分解（不含雀头）。"""

    __slots__ = ("melds",)

    def __init__(self, melds):
        self.melds = melds

    def __len__(self):
        return len(self.melds)

    def __repr__(self):
        from .tiles import tile_name
        ms = []
        for kind, t in self.melds:
            if kind == CHI:
                ms.append(f"{tile_name(t)}{tile_name(t+1)}{tile_name(t+2)}")
            else:
                ms.append(tile_name(t) * 3)
        return "Decomp[" + " ".join(ms) + "]"


def find_pair(counts):
    """返回 (pair_tile or None, remaining_counts)。"""
    for t in range(NUM_TILES):
        if counts[t] >= 2:
            c = list(counts)
            c[t] -= 2
            return t, c
    return None, None


def is_standard_complete(counts, meld_count):
    """已归并手牌（含副露牌），判断是否为标准型和了。
    meld_count = 副露组数 + 1（本组面子）。总组数须为 4。
    """
    # 本组需要凑 (4 - meld_count) 组面子 + 1 雀头
    need_melds = 4 - meld_count
    if need_melds < 0:
        return False
    c = list(counts)
    # 先尝试每个雀头
    for pair_tile in range(NUM_TILES):
        if c[pair_tile] < 2:
            continue
        c[pair_tile] -= 2
        for dec in enumerate_standard(c, need_melds):
            if dec is not None:
                c[pair_tile] += 2
                return True
        c[pair_tile] += 2
    return False


def enumerate_full(counts, meld_count):
    """返回所有完整的 (Decomposition, pair_tile)。"""
    results = []
    need_melds = 4 - meld_count
    if need_melds < 0:
        return results
    c = list(counts)
    for pair_tile in range(NUM_TILES):
        if c[pair_tile] < 2:
            continue
        c[pair_tile] -= 2
        for dec in enumerate_standard(c, need_melds):
            results.append((dec, pair_tile))
        c[pair_tile] += 2
    return results


def enumerate_all_with_melds(concealed_counts, melds):
    """给定暗牌计数与副露列表，枚举所有完整的和了分解（含副露）。

    concealed_counts: 34 维，仅暗牌
    melds: list of (kind, tile, concealed_bool)  副露单元

    返回 list of (list_of_meld_tuples, pair_tile)
    其中 list_of_meld_tuples 含暗牌分解的 meld + 副露 meld。
    """
    meld_count = len(melds)
    need = 4 - meld_count
    if need < 0:
        return []
    # 把副露的牌也并入计数，枚举统一分解
    merged = list(concealed_counts)
    for kind, t, concealed in melds:
        size = 4 if kind == KAN else 3
        merged[t] += size
        if kind == CHI:
            merged[t + 1] += 1
            merged[t + 2] += 1
    out = []
    c = list(merged)
    for pair_tile in range(NUM_TILES):
        if c[pair_tile] < 2:
            continue
        c[pair_tile] -= 2
        for dec in enumerate_standard(c, need):
            out.append((list(dec.melds) + list(melds), pair_tile))
        c[pair_tile] += 2
    return out


# ---------------------------------------------------------------------------
# 特殊和了型（RULES.md §6.2 / §6.3）
# ---------------------------------------------------------------------------

def is_chiitoitsu(counts, meld_count):
    """七对子：7 个对子，门前限定（meld_count == 0），杠不算破门。
    counts 含全部手牌 + 副露牌；副露必须为 0。
    """
    if meld_count != 0:
        return False
    pairs = 0
    t = 0
    for i in range(NUM_TILES):
        if counts[i] == 0:
            continue
        if counts[i] % 2 == 0:
            pairs += counts[i] // 2
        else:
            return False  # 奇数张无法成对（七对子不容忍单张）
        t += counts[i]
    return pairs == 7


def chiitoi_shape(counts):
    """返回七对子的对子列表，若不是七对子返回 None。"""
    if not is_chiitoitsu(counts, 0):
        return None
    pairs = []
    for i in range(NUM_TILES):
        if counts[i]:
            pairs.extend([i] * (counts[i] // 2))
    return pairs


def is_kokushi(counts, meld_count):
    """国士无双十三幺：13 种幺九牌齐全 + 其中一种成对（共 14 张）。

    本规范不强制门前（§17-A6b 自决项），故 meld_count 允许非 0。
    """
    # 快速否定：国士必须恰好 14 张，且全部是 13 种幺九牌。
    # 这两个检查是 O(1) 的，能挡掉绝大多数调用（实测省掉 ~95% 时间）。
    n = 0
    for i in range(NUM_TILES):
        c = counts[i]
        if c:
            n += c
            if c > 2 or i not in _Y9_SET:
                return False
    return n == 14


def kokushi_pair(counts):
    """国士无双的雀头（对子牌），若不是国士返回 None。"""
    if not is_kokushi(counts, 0):
        return None
    for t in TERMINALS_AND_HONORS:
        if counts[t] == 2:
            return t
    return None


def is_complete(counts, meld_count):
    """综合判定：标准型 / 七对子 / 国士无双，任一成立即为和了牌型。"""
    return (
        is_standard_complete(counts, meld_count)
        or is_chiitoitsu(counts, meld_count)
        or is_kokushi(counts, meld_count)
    )
