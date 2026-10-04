"""算分汇总：把一副和了分解变成 (yaku 列表, han, fu, 点数)。

对应 RULES.md §7（符/番）、§8（结算表）、§9（和了流程）。

本模块只回答「哪些役成立、符多少、值多少点」。
它不回答「这手牌好不好」「该打哪张」—— 那是策略网络的事（RULES.md §15）。

数据流
------
    decomposition (melds + pair) + 场面标志
      -> 役满判定（短路）
      -> 普通役判定 -> 番数汇总
      -> 符（跳符 + 符翻）
      -> 查表得点
"""

from __future__ import annotations

from . import ruleset as RS
from . import yaku as Y
from .tiles import NUM_TILES, is_honor, is_terminal
from .hand import CHI, PONG, KAN

MENZEN_FU = 20
MELD_FU = 10
KAN_FU = 8
HAKU, NORTH = 31, 30


class ScoreResult:
    """一副牌的完整算分结果。

    points = 和了方总共收的点数（不含跳满以上；跳满以上见 limit_points）
    """

    __slots__ = ("yakus", "han", "fu", "points", "yakuman_mult",
                 "is_menzen", "is_valid")

    def __init__(self, yakus, han, fu, points, yakuman_mult, is_menzen):
        self.yakus = tuple(yakus)
        self.han = han
        self.fu = fu
        self.points = points
        self.yakuman_mult = yakuman_mult
        self.is_menzen = is_menzen
        self.is_valid = bool(yakus) or yakuman_mult > 0

    def __repr__(self):
        return (f"Score[{'+'.join(self.yakus) or 'no-yaku'} "
                f"han={self.han} fu={self.fu} "
                f"yakuman={self.yakuman_mult} pts={self.points}]")


# ---------------------------------------------------------------------------
# 符
# ---------------------------------------------------------------------------

def _pon_kan_fu(tile, is_kan, is_open):
    """刻子/杠的符。对照 MahjongRepository/mahjong 的 _PON_KAN_FU_TABLE。

    中张:  暗刻 4 / 明刻 2 / 暗杠 16 / 明杠 8
    幺九:  暗刻 8 / 明刻 4 / 暗杠 32 / 明杠 16
    """
    terminal = is_honor(tile) or is_terminal(tile)
    if is_kan:
        return (16 if is_open else 32) if terminal else (8 if is_open else 16)
    return (4 if is_open else 8) if terminal else (2 if is_open else 4)


def compute_fu(melds, pair_tile, is_menzen, seat_wind, round_wind,
               is_pinfu=False, is_tsumo=False, win_tile=None,
               win_meld=None, chiitoi=False):
    """计算符。

    对照实现（2026-10-04 外部核验，源：MahjongRepository/mahjong 2.0.0
    FuCalculator.calculate_fu）：

    1. 刻子/杠符按表（暗/明 × 中张/幺九 × 刻/杠）
    2. 雀头符：场风/自风/白 各 2（可叠加）
    3. 边张 2 / 嵌张 2 / 单骑 2（只对门前）
    4. 自摸额外 2 符（符已为 0 时不给）
    5. 底符：门前放铳 30；副露或自摸 20
    6. 末尾统一向上取整到 10
    """
    if chiitoi:
        return 25

    is_open_hand = not is_menzen
    fu = 0

    for m in melds:
        kind, t = m[0], m[1]
        concealed = len(m) > 2 and m[2]
        if kind == PONG:
            fu += _pon_kan_fu(t, False, not concealed)
        elif kind == KAN:
            fu += _pon_kan_fu(t, True, not concealed)

    valued = 0
    if pair_tile == HAKU or pair_tile == NORTH:
        valued += 1
    if seat_wind is not None and pair_tile == seat_wind:
        valued += 1
    if round_wind is not None and pair_tile == round_wind:
        valued += 1
    fu += 2 * valued

    if is_menzen and win_meld is not None and win_tile is not None:
        if win_meld[0] == H.PAIR:
            if win_tile == win_meld[1]:
                fu += 2
        elif win_meld[0] == H.CHI:
            start = win_meld[1]
            rank = start % 9
            if win_tile == start + 2 and rank == 0:
                fu += 2
            elif win_tile == start and rank == 6:
                fu += 2
            elif win_tile == start + 1:
                fu += 2

    if is_tsumo and fu > 0:
        fu += 2

    base = 30 if (is_menzen and not is_tsumo) else 20
    fu += base

    if is_open_hand and fu == 0:
        fu += 2

    # --- 向上取整到 10 ---
    # 注意：门前平和**不需要**再翻到 40。外部核验（2026-10-04，
    # MahjongRepository/mahjong 2.0.0 FuCalculator）显示门前平和放铳
    # 就是 30 符——底符本身已是 30。原实现误加一次「符翻」顶到 40。
    fu = (fu + 9) // 10 * 10
    return fu

# ---------------------------------------------------------------------------
# 役的番数表
# ---------------------------------------------------------------------------

YAKU_HAN = {
    "断幺九": 1,
    "立直": 1,
    "一发": 1,
    "门前清": 1,
    "一盃口": 1,
    "平和": 1,
    "一飜": 1,
    "对对和": 1,
    "三暗刻": 1,
    "三色同顺": 2,
    "三色同刻": 2,
    "混全带幺九": 2,
    "纯全带幺九": 3,
    "二盃口": 2,
    "七对子": 2,
    "混老头": 2,
    "三暗刻单骑": 2,
    "双立直": 2,
    "清一色": 6,
    "混一色": 3,
    "字一色": 13,
}

YAKUMAN_1 = 4     # 役满
YAKUMAN_2 = 6     # 双倍役满
YAKUMAN_HAN = {4: 13, 6: 26}   # 记番时用的近似值


# ---------------------------------------------------------------------------
# 役满判定
# ---------------------------------------------------------------------------

def _count_kan(melds):
    return sum(1 for m in melds if m[0] == KAN)


def yakuman_multiplier(melds, pair_tile, counts, meld_count):
    """返回役满倍数（0 = 非役满）。"""
    mult = 0

    if Y.is_kokushi(counts, meld_count):
        mult = max(mult, YAKUMAN_1)

    if len(melds) == 4 and all(m[0] != CHI for m in melds) and \
            all(len(m) > 2 and m[2] for m in melds):
        mult = max(mult, YAKUMAN_1)   # 四暗刻（含单骑）

    if Y.is_daisuushii(melds, pair_tile):
        mult = max(mult, YAKUMAN_2)

    if Y.is_suushii(melds, pair_tile):
        mult = max(mult, YAKUMAN_1)

    if Y.is_daisangen(melds, pair_tile):
        mult = max(mult, YAKUMAN_1)

    if all(is_honor(t) for t in _all_tiles(melds, pair_tile)):
        mult = max(mult, YAKUMAN_1)   # 字一色

    if _count_kan(melds) == 4:
        mult = max(mult, YAKUMAN_1)   # 四杠

    return mult


def _all_tiles(melds, pair_tile):
    out = []
    for m in melds:
        kind, t = m[0], m[1]
        if kind == CHI:
            out.extend((t, t + 1, t + 2))
        elif kind == PONG:
            out.extend((t, t, t))
        else:
            out.extend((t, t, t, t))
    out.extend((pair_tile, pair_tile))
    return out


# ---------------------------------------------------------------------------
# 普通役
# ---------------------------------------------------------------------------

def _detect_yakus(melds, pair_tile, counts, meld_count, is_menzen,
                  is_tsumo, is_riichi, is_ippatsu, is_double_riichi,
                  seat_wind, round_wind, is_chiitoi_shape, tenpai):
    """返回成立的役名列表。"""
    yakus = []

    if is_riichi:
        yakus.append("立直")
    if is_double_riichi:
        yakus.append("双立直")
    if is_ippatsu:
        yakus.append("一发")
    if is_menzen:
        yakus.append("门前清")

    if is_chiitoi_shape:
        yakus.append("七对子")
    else:
        if Y.is_tanyao(melds, pair_tile):
            yakus.append("断幺九")
        if all(m[0] != CHI for m in melds):
            yakus.append("对对和")
        if Y.is_iippekou(melds, is_menzen, pair_tile):
            yakus.append("二盃口")
        elif Y.is_ittou(melds, is_menzen):
            yakus.append("一盃口")
        if Y.is_pinfu(melds, pair_tile, is_menzen, seat_wind, round_wind):
            yakus.append("平和")
        if Y.is_sanankou(melds):
            yakus.append("三暗刻")
        if Y.is_sanshoku(melds):
            yakus.append("三色同顺")
        if Y.is_itttsu(melds):
            yakus.append("三色同刻")
        if Y.is_junchan(melds, pair_tile, meld_count):
            yakus.append("混全带幺九")
        elif Y.is_chunchan(melds, pair_tile, meld_count):
            yakus.append("纯全带幺九")
        if _is_junchan_or_koutou(melds, pair_tile):
            yakus.append("混老头")
        if _is_all_honors(melds, pair_tile):
            yakus.append("字一色")
        elif Y.is_honitsu(melds, pair_tile):
            yakus.append("混一色")
        elif Y.is_chinitsu(melds, pair_tile):
            yakus.append("清一色")

    return yakus


def _is_all_honors(melds, pair_tile):
    return all(is_honor(t) for t in _all_tiles(melds, pair_tile))


def _is_junchan_or_koutou(melds, pair_tile):
    """混老头（koutou）：全老头牌 + 字牌。"""
    has_honor = False
    for m in melds:
        kind, t = m[0], m[1]
        if kind == CHI:
            return False
        if is_honor(t):
            has_honor = True
        elif not is_terminal(t):
            return False
    if is_honor(pair_tile):
        has_honor = True
    elif not is_terminal(pair_tile):
        return False
    return has_honor


# ---------------------------------------------------------------------------
# 顶层算分
# ---------------------------------------------------------------------------

def evaluate(melds, pair_tile, counts, meld_count, is_menzen,
             is_tsumo, is_riichi, is_ippatsu, is_double_riichi,
             seat_wind, round_wind, is_chiitoi_shape,
             dora_count=0, aka_count=0, ura_dora_count=0,
             is_dealer=False, tenpai=False):
    """算一副和了的分。

    参数
    ----
    melds: [(kind, tile) 或 (kind, tile, concealed_bool), ...]，共 4 组
    pair_tile: 雀头牌 id
    counts: 34 维完整计数（含副露牌与雀头），用于七对/国士判定
    meld_count: 副露组数
    is_menzen: 和了者是否门前（含立直）
    is_tsumo: 是否自摸
    is_riichi / is_ippatsu / is_double_riichi: 场面标志
    seat_wind / round_wind: 自风、场风 tile id
    is_chiitoi_shape: 是否七对子型
    dora_count / aka_count / ura_dora_count: 宝牌/赤/里宝牌计数

    返回 ScoreResult。无役且非役满时 is_valid=False。
    """
    yakuman = yakuman_multiplier(melds, pair_tile, counts, meld_count)
    if yakuman:
        pts = RS.limit_points(yakuman)
        names = ["役满" if yakuman == YAKUMAN_1 else "双倍役满"]
        return ScoreResult(names, YAKUMAN_HAN.get(yakuman, 13), 0, pts,
                           yakuman, is_menzen)

    is_pinfu = (not is_chiitoi_shape
                and Y.is_pinfu(melds, pair_tile, is_menzen, seat_wind, round_wind))
    yakus = _detect_yakus(melds, pair_tile, counts, meld_count, is_menzen,
                          is_tsumo, is_riichi, is_ippatsu, is_double_riichi,
                          seat_wind, round_wind, is_chiitoi_shape, tenpai)

    han = 0
    for y in yakus:
        han += YAKU_HAN.get(y, 0)
    han += dora_count + aka_count + ura_dora_count

    if han == 0:
        fu = compute_fu(melds, pair_tile, is_menzen, seat_wind, round_wind,
                        is_pinfu=False)
        return ScoreResult([], 0, fu, 0, 0, is_menzen)

    # 符 20 / 25 需要翻符才成役
    if is_chiitoi_shape:
        fu = 25
    else:
        fu = compute_fu(melds, pair_tile, is_menzen, seat_wind, round_wind,
                        is_pinfu=is_pinfu)

    # 符翻：符 20 的番必须 >= 2 翻才成立；符 25 的 >= 2 翻
    if fu == 20 and han < 2 and not is_riichi:
        return ScoreResult([], 0, fu, 0, 0, is_menzen)
    if fu == 25 and han < 2:
        return ScoreResult([], 0, fu, 0, 0, is_menzen)

    han = min(han, 5)   # 5 番封顶查表

    if is_chiitoi_shape:
        fu = 25
    key_fu = 25 if is_chiitoi_shape else fu

    pts = RS.han_fu_points(key_fu, han, is_dealer=is_dealer)
    return ScoreResult(yakus, han, key_fu, pts, 0, is_menzen)
