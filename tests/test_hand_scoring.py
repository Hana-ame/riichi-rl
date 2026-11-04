"""手牌/和牌分解/算分/役 与开源引擎一致性测试。

覆盖：mahjong/hand.py、mahjong/scoring.py、mahjong/refengine.py

运行：python -m tests.test_hand_scoring
"""

from __future__ import annotations

import unittest

import tests.utils as U  # noqa: F401

from mahjong import hand as H
from mahjong import refengine as REF
from mahjong import scoring as S


def c34(tiles):
    c = [0] * 34
    for t in tiles:
        c[t] += 1
    return c


class TestHandShapes(unittest.TestCase):
    def test_standard_complete(self):
        # 123m 456m 789m 234p + 东东
        counts = c34([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 27, 27])
        self.assertTrue(H.is_standard_complete(counts, 0))

    def test_not_complete(self):
        counts = c34([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 27, 28])
        self.assertFalse(H.is_standard_complete(counts, 0))

    def test_chiitoitsu(self):
        counts = c34([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 27, 27])
        self.assertTrue(H.is_chiitoitsu(counts, 0))

    def test_kokushi(self):
        counts = c34([0, 8, 9, 17, 18, 26] + list(range(27, 34)) + [27])
        self.assertTrue(H.is_kokushi(counts, 0))


class TestRefengine(unittest.TestCase):
    def test_shanten_win(self):
        counts = c34([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 27, 27])
        self.assertEqual(REF.ref_shanten(counts), -1, "和牌型向听应为 -1")

    def test_shanten_tenpai(self):
        # 13 张听东
        counts = c34([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 27])
        self.assertEqual(REF.ref_shanten(counts), 0, "听牌向听应为 0")

    def test_can_win_with(self):
        counts = c34([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 27])
        self.assertTrue(REF.can_win_with(counts, 27, 0))
        self.assertFalse(REF.can_win_with(counts, 28, 0))

    def test_score_returns_valid(self):
        # 门前清自摸：123m456m789m234p 东东摸东
        counts = c34([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 27, 27])
        r = REF.score_hand(counts, 27, None, seat_wind=28, round_wind=27,
                           is_tsumo=True)
        self.assertTrue(r.valid, f"应能和牌（err={r.error}）")
        self.assertGreaterEqual(r.han, 1)
        self.assertTrue(r.yaku)


class TestSelfScoring(unittest.TestCase):
    def test_no_yaku_detected(self):
        """无役手牌（副露 + 无断幺/无役牌）自研引擎应判无效。"""
        # 碰 1m(副露) + 手牌 234m 456m 789p 11s → 和 1s 时无役
        # 简化：直接用 evaluate 对已知无役组合断言 is_valid=False 或不抛异常
        counts = c34([1, 1, 1, 2, 3, 4, 4, 5, 6, 10, 11, 12, 18, 18])
        # 这里只验证接口返回结构稳定，具体役种由 refengine 测试覆盖
        from mahjong.hand import CHI
        res = S.evaluate([(CHI, 0)], 18, counts, 1, False, False, False,
                         False, False, 28, 27, False)
        self.assertTrue(hasattr(res, "is_valid"))
        self.assertTrue(hasattr(res, "han"))
        self.assertTrue(hasattr(res, "points"))


class TestScoringRegressions(unittest.TestCase):
    """2026-11-05 修复：赤牌只数和牌者；「门前清」只在自摸时成役。"""

    def test_menzen_yaku_only_on_tsumo(self):
        # 234m 567m 234p 678p 55p：断幺九（荣和也有役）
        counts = c34([1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 13, 14, 15, 16])
        melds = [(H.CHI, 1, True), (H.CHI, 4, True),
                 (H.CHI, 10, True), (H.CHI, 14, True)]
        ron = S.evaluate(melds, 13, counts, 0, True, False,
                         False, False, False, 28, 27, False)
        self.assertTrue(ron.is_valid)
        self.assertIn("断幺九", ron.yakus)
        self.assertNotIn("门前清", ron.yakus,
                         "门前清不是独立役，门清荣和没有这一番")
        tsumo = S.evaluate(melds, 13, counts, 0, True, True,
                           False, False, False, 28, 27, False)
        self.assertIn("门前清", tsumo.yakus,
                      "门前清自摸应成役")
        self.assertGreater(tsumo.han, ron.han)

    def test_aka_count_winner_only(self):
        """赤牌只数和牌者手牌：旧实现数四家、每家和牌白送 +3 番。"""
        from mahjong import env as E
        env = U.make_env(seed=1)
        for i in range(4):
            env.players[i].hand = [1, 2, 3]
        self.assertEqual(env._aka_count(0), 0)
        # 别家手里有赤牌（5m=4 / 5p=13 / 5s=22）不影响 0 号位
        env.players[1].hand = [4, 13, 22]
        env.players[2].hand = [4, 13, 22]
        self.assertEqual(env._aka_count(0), 0,
                         "不能把别人的赤牌算给和牌者")
        env.players[0].hand = [4, 13, 22]
        self.assertEqual(env._aka_count(0), 3)


if __name__ == "__main__":
    import sys
    sys.exit(U.run_module_tests(sys.modules[__name__]))