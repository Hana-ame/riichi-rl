"""2026-11-27 引擎/封装/外部Elo修复回归测试。"""
from __future__ import annotations
import random
import unittest
import tests.utils as U  # noqa: F401

from mahjong import env as E
from mahjong import scoring as S
from mahjong.shanten import shanten, standard_shanten
from mahjong.tiles import tile_name
from arena_elo import _rule_override
import external_elo_ext as EXT


def c34(names):
    m = {tile_name(i): i for i in range(34)}
    c = [0] * 34
    for n in names:
        c[m[n]] += 1
    return c


class TestEngineFixes20261127(unittest.TestCase):
    def test_shanten_not_false_tenpai(self):
        # 456m 888m 556p 1223s 是 1 向听，绝不是 0 向听
        c = c34(["4m","5m","6m","8m","8m","8m","5p","5p","6p","1s","2s","2s","3s"])
        self.assertEqual(standard_shanten(c, 0), 1)
        self.assertEqual(shanten(c, 0), 1)

    def test_scoring_limit_hands_and_dragon_fu(self):
        # 5 番满贯：庄 12000 / 子 8000；6 番跳满：庄 18000 / 子 12000
        from mahjong.hand import CHI
        # 234m 567m 345p 456s + 55s（无三色）：立直1 + 断幺1 + 平和1 + 宝牌2 = 5 番满贯
        melds = [(CHI, 1, True), (CHI, 4, True), (CHI, 11, True), (CHI, 21, True)]
        counts = c34(["2m","3m","4m","5m","6m","7m","3p","4p","5p","4s","5s","6s","5s","5s"])
        r5_d = S.evaluate(melds, 22, counts, 0, True, False, True, False, False, 27, 27, False, dora_count=2, is_dealer=True)
        r5_c = S.evaluate(melds, 22, counts, 0, True, False, True, False, False, 28, 27, False, dora_count=2, is_dealer=False)
        self.assertEqual(r5_d.points, 12000)
        self.assertEqual(r5_c.points, 8000)
        r6_d = S.evaluate(melds, 22, counts, 0, True, False, True, False, False, 27, 27, False, dora_count=3, is_dealer=True)
        self.assertEqual(r6_d.han, 6)
        self.assertEqual(r6_d.points, 18000)
        # 雀头符：发/中加 2 符，北风非场风/自风不加符
        fu_hatsu = S.compute_fu(melds, 32, True, 28, 27)
        fu_north = S.compute_fu(melds, 30, True, 28, 27)
        self.assertGreater(fu_hatsu, fu_north)

    def test_kuikae_forbidden_after_pon(self):
        env = E.MahjongEnv(seed=42)
        env.start_turn()
        p = env.players[1]
        p.hand = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 24, 24, 24]  # 暗刻 7s
        env.last_discard = 24
        env.last_discard_player = 0
        env.players[0].river.append(24)
        env._do_pon(1, 24)
        legal = env.legal_actions(1)
        disc_tiles = {a.tile for a in legal if a.type == E.ACT_DISCARD}
        self.assertNotIn(24, disc_tiles, "碰 7s 后当巡禁止打出 7s（食替）")

    def test_rule_override_does_not_force_riichi(self):
        legal = [E.Action(E.ACT_DISCARD, 0), E.Action(E.ACT_RIICHI, 0)]
        self.assertIsNone(_rule_override(legal))
        legal_win = [E.Action(E.ACT_DISCARD, 0), E.Action(E.ACT_TSUMO)]
        self.assertEqual(_rule_override(legal_win).type, E.ACT_TSUMO)

    def test_reselect_includes_bottom_half(self):
        roster = [f"m{i}" for i in range(8)]
        ratings = {f"m{i}": 1600.0 - i * 20 for i in range(8)}
        rng = random.Random(2026)
        active = EXT.reselect(ratings, roster, rng)
        self.assertEqual(len(active), 4)
        self.assertTrue(any(n in roster[4:] for n in active), "reselect 必须包含后一半随机替换")


if __name__ == "__main__":
    unittest.main()
