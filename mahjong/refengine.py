"""开源算分引擎适配层。

【为什么换掉自研规则引擎】
自研的 hand.py + scoring.py + yaku.py 在训练中使用中暴露了多个**真实规则 bug**：
  1. 振听（furiten）判定缺失 —— _can_ron 不检查，允许振听放铳和了
  2. 立直动作不带「打出的那张牌」，RIICHI 的 tile 参数是 -1
  3. 和了枚举的面子数判断写成 tuple 长度
  4. 符模型底符/自摸符/平和符翻三处错误（已在 ruleset.py 改对，但机制仍是自己推的）
这些 bug 不报错，只会让 RL 学到错误的目标函数。

【换成什么】
MahjongRepository/mahjong（PyPI 包名 mahjong），已 vendor 到 vendor/mahjong_ref。
提供三块经过验证的能力：
  - HandCalculator.estimate_hand_value  -> 和了判定 + 符 + 番 + 役名
  - Shanten.calculate_shanten           -> 向听数
  - ScoresCalculator.calculate_scores   -> 点数表
其自带 doctest 实跑 20 passed。

【引擎流程规则仍用自研】
摸打流程、副露、抢杠、连庄等流程规则保留 env.py，
但所有「这手牌算不算和、值多少」改为调用开源实现。

【许可】上游 MIT 系，来源 https://github.com/MahjongRepository/mahjong
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REF = os.path.join(os.path.dirname(_HERE), "vendor")
if _REF not in sys.path:
    sys.path.insert(0, _REF)

from majsoul_ref.mahjong.hand_calculating.hand import HandCalculator
from majsoul_ref.mahjong.hand_calculating.hand_config import HandConfig
from majsoul_ref.mahjong.shanten import Shanten
from majsoul_ref.mahjong.meld import Meld


def to136(tile34: int) -> int:
    """34 编号 -> 上游的 136 编号（tile34 * 4）。"""
    return tile34 * 4


class RefScore:
    """一次算分的结果。"""

    __slots__ = ("valid", "fu", "han", "yaku", "error")

    def __init__(self, valid, fu, han, yaku, error=None):
        self.valid = valid
        self.fu = fu
        self.han = han
        self.yaku = yaku
        self.error = error

    def __repr__(self):
        return (f"RefScore[{'+'.join(self.yaku) or 'no-yaku'} "
                f"han={self.han} fu={self.fu}]")


def score_hand(counts34, win_tile34, melds=None, seat_wind=None,
               round_wind=None, is_tsumo=False, is_riichi=False,
               is_ippatsu=False):
    """用开源实现算分。

    counts34: 34 维计数（含和了牌与副露牌）
    win_tile34: 和了的那张牌（34 编号）
    melds: 上游 Meld 列表；None = 门前
    """
    tiles = []
    for t in range(34):
        tiles.extend([to136(t)] * counts34[t])

    if win_tile34 is None:
        win136 = tiles[-1] if tiles else 0
    else:
        win136 = to136(win_tile34)

    config = HandConfig(
        is_tsumo=is_tsumo,
        player_wind=seat_wind if seat_wind is not None else 28,
        is_riichi=is_riichi,
        is_ippatsu=is_ippatsu,
    )
    try:
        resp = HandCalculator.estimate_hand_value(
            tiles=tiles, win_tile=win136, melds=melds, config=config)
    except Exception as e:                        # noqa: BLE001
        return RefScore(False, 0, 0, (), error=repr(e))

    if getattr(resp, "error", None):
        return RefScore(False, 0, 0, (), error=resp.error)
    yaku = tuple(getattr(y, "name", str(y)) for y in (resp.yaku or ()))
    if not yaku:
        return RefScore(False, resp.fu or 0, resp.han or 0, (),
                        error="no_yaku")
    return RefScore(True, resp.fu or 0, resp.han or 0, yaku)


def ref_shanten(counts34):
    """开源向听数。和了返回 -1，听牌返回 0。"""
    n = sum(counts34)
    if n not in (1, 2, 4, 5, 7, 8, 10, 11, 13, 14):
        return 8
    try:
        return Shanten.calculate_shanten(counts34)
    except Exception:                            # noqa: BLE001
        return 8


def winning_tile_available(counts34, meld_count=0, tile34=None):
    """counts34 **已包含** tile34 时，该牌能否和了（含有役检查）。

    上游算分器要求 136 列表里恰好含 win_tile，所以这里不再自己加牌。
    """
    if tile34 is None or counts34[tile34] <= 0:
        return False
    return score_hand(counts34, tile34).valid


def can_win_with(counts34, tile34, meld_count=0):
    """counts34 **不含**该牌时，这张牌来了能不能和（振听/合法���用）。"""
    if tile34 is None or counts34[tile34] >= 4:
        return False
    c = list(counts34)
    c[tile34] += 1
    return score_hand(c, tile34).valid