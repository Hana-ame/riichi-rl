"""役（yaku）判定与汇总算分。

对应 RULES.md §7.4、§8。本模块只回答「哪些役成立」「符多少」「点多少」，
不含任何策略判断（RULES.md §15）。
"""

from __future__ import annotations

from . import ruleset as RS
from .tiles import (
    NUM_TILES, HAKU, is_honor, is_terminal, TERMINALS_AND_HONORS,
)
from .hand import CHI, PONG, KAN, PAIR


# ---------------------------------------------------------------------------
# 单个役的判定函数
# ---------------------------------------------------------------------------

def _meld_tiles(m):
    kind, t = m[0], m[1]
    if kind == CHI:
        return (t, t + 1, t + 2)
    if kind == PONG:
        return (t, t, t)
    return (t, t, t, t)


def is_tanyao(melds, pair_tile):
    """断幺九：全部为 2-8 的数牌。"""
    for m in melds:
        for t in _meld_tiles(m):
            if is_terminal(t) or is_honor(t):
                return False
    return not (is_terminal(pair_tile) or is_honor(pair_tile))


def is_pinfu(melds, pair_tile, is_menzen, seat_wind, round_wind,
             is_chiitoi_shape=False):
    """平和：门前 + 全顺子 + 雀头非自风/场风/白/北。"""
    if not is_menzen or is_chiitoi_shape:
        return False
    if any(m[0] != CHI for m in melds):
        return False
    for bad in (HAKU, 30):
        if pair_tile == bad:
            return False
    if seat_wind is not None and pair_tile == seat_wind:
        return False
    if round_wind is not None and pair_tile == round_wind:
        return False
    return True


def is_iippekou(melds, is_menzen, pair_tile):
    """二盃口：门前两副完全相同的顺子。"""
    if not is_menzen:
        return False
    seqs = [m[1] for m in melds if m[0] == CHI]
    if len(seqs) != 4:
        return False
    return len(set(seqs)) == 3


def is_ittou(melds, is_menzen):
    """一盃口：门前三副相同顺子 + 一副不同顺子。"""
    if not is_menzen:
        return False
    seqs = [m[1] for m in melds if m[0] == CHI]
    if len(seqs) != 4:
        return False
    from collections import Counter
    c = Counter(seqs)
    return max(c.values()) == 3


def is_sanshoku(melds):
    """三色同顺：同起点（或同数位）的三门顺子。"""
    by_rank = {}
    for m in melds:
        if m[0] != CHI:
            continue
        t = m[1]
        rank = t % 9
        by_rank.setdefault(rank, set()).add(t // 9)
    return any(len(v) == 3 for v in by_rank.values())


def is_itttsu(melds):
    """三色同刻：同牌位的三门刻子/杠。"""
    by_rank = {}
    for m in melds:
        t = m[1]
        if is_honor(t):
            continue
        rank = t % 9
        by_rank.setdefault(rank, set()).add(t // 9)
    return any(len(v) == 3 for v in by_rank.values())


def is_toitoi(melds, pair_tile):
    """对对和：无顺子。"""
    return all(m[0] != CHI for m in melds)


def is_sanankou(melds):
    """三暗刻：3 个暗刻/暗杠。"""
    n = 0
    for m in melds:
        if len(m) > 2 and m[2] and m[0] in (PONG, KAN):
            n += 1
    return n == 3


def is_suuankou(melds, pair_tile, waiting_single):
    """四暗刻（+ 单骑）。"""
    if any(m[0] == CHI for m in melds):
        return False
    n = 0
    for m in melds:
        if len(m) > 2 and m[2]:
            n += 1
        else:
            # 明刻/明杠不算暗，但四暗刻要求全暗
            return False
    return n == 4


def is_chiitoitsu(counts):
    pairs = 0
    for i in range(NUM_TILES):
        if counts[i] % 2:
            return False
        pairs += counts[i] // 2
    return pairs == 7


def is_honitsu(melds, pair_tile):
    suits, honors = set(), False
    for m in melds:
        for t in _meld_tiles(m):
            if is_honor(t):
                honors = True
            else:
                suits.add(t // 9)
    if is_honor(pair_tile):
        honors = True
    else:
        suits.add(pair_tile // 9)
    return len(suits) == 1 and honors


def is_chinitsu(melds, pair_tile):
    suits = set()
    for m in melds:
        for t in _meld_tiles(m):
            if is_honor(t):
                return False
            suits.add(t // 9)
    if is_honor(pair_tile):
        return False
    suits.add(pair_tile // 9)
    return len(suits) == 1


def _junchan_shape(melds, pair_tile):
    """返回 True 表示每组都含幺九牌（雀头不要求）。"""
    for m in melds:
        if m[0] == CHI:
            r = m[1] % 9
            if r not in (0, 6):
                return False
        else:
            if not (is_terminal(m[1]) or is_honor(m[1])):
                return False
    return True


def is_junchan(melds, pair_tile, meld_count):
    """混全带幺九：每组都含幺九牌，且至少一组是副露。"""
    return meld_count > 0 and _junchan_shape(melds, pair_tile)


def is_chunchan(melds, pair_tile, meld_count):
    """纯全带幺九：每组都含幺九牌，门前，且雀头是中张。"""
    if meld_count:
        return False
    return _junchan_shape(melds, pair_tile) and not is_terminal(pair_tile) \
        and not is_honor(pair_tile)





def is_kokushi(counts, meld_count, menzen_only=False):
    if sum(counts) != 14:
        return False
    if menzen_only and meld_count:
        return False
    pair_found = False
    present = 0
    for t in TERMINALS_AND_HONORS:
        if counts[t] == 0:
            continue
        if counts[t] > 2:
            return False
        if counts[t] == 2:
            if pair_found:
                return False
            pair_found = True
        present += 1
    for t in range(NUM_TILES):
        if t not in TERMINALS_AND_HONORS and counts[t]:
            return False
    return present == 13 and pair_found


def is_daisangen(melds, pair_tile):
    """大三元：三元牌刻子/杠各 1 组。"""
    dragons = (HAKU, 32, 33)
    got = set()
    for m in melds:
        if m[0] != CHI and m[1] in dragons:
            got.add(m[1])
    return len(got) == 3


def is_suushii(melds, pair_tile):
    """小四喜：3 风刻子 + 1 风雀头。"""
    winds = {27, 28, 29, 30}
    got = set()
    for m in melds:
        if m[0] != CHI and m[1] in winds:
            got.add(m[1])
    has_pair_wind = pair_tile in winds
    return len(got) == 3 and has_pair_wind


def is_daisuushii(melds, pair_tile):
    """大四喜：4 风刻子/杠。"""
    winds = {27, 28, 29, 30}
    got = {m[1] for m in melds if m[0] != CHI and m[1] in winds}
    return len(got) == 4


def is_ryuukyoku_possible(counts, meld_count):
    """七对听牌：13 张里 6 对 + 1 单张（单张可成对）。"""
    if meld_count:
        return False
    singles = 0
    pairs = 0
    for i in range(NUM_TILES):
        if counts[i] == 0:
            continue
        if counts[i] == 1:
            singles += 1
        elif counts[i] % 2 == 0:
            pairs += counts[i] // 2
        else:
            return False
    return pairs == 6 and singles == 1
