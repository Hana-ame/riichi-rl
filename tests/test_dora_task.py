"""P4：dora 额外任务测试（和牌且宝牌/赤/里宝≥1 番）。"""

from __future__ import annotations

import unittest

from mahjong import aux_tasks as AUX
from mahjong.env import GameResult


def _res(winner=0, dora=0, aka=0, ura=0):
    return GameResult(
        result_type="tsumo", winner=winner, from_player=None,
        points_delta=[0, 0, 0, 0], scores=[25000] * 4, yaku=("立直",),
        han=1, fu=30, tenpai=[False] * 4, log=[], turns=10,
        dora_count=dora, aka_count=aka, ura_dora_count=ura)


class TestDoraTask(unittest.TestCase):
    def test_dora_task_registered(self):
        self.assertIn("dora", AUX.TASKS)
        self.assertIn("dora", AUX.STATIC_TASKS)

    def test_dora_completed_with_any_dora(self):
        env = type("E", (), {"players": [type("P", (), {"riichi": False})()
                                        for _ in range(4)]})()
        done = AUX.static_tasks(env, _res(dora=1), 0)
        self.assertTrue(done["dora"])
        done2 = AUX.static_tasks(env, _res(aka=1), 0)
        self.assertTrue(done2["dora"])
        done3 = AUX.static_tasks(env, _res(ura=1), 0)
        self.assertTrue(done3["dora"])

    def test_no_dora_no_bonus(self):
        env = type("E", (), {"players": [type("P", (), {"riichi": False})()
                                        for _ in range(4)]})()
        done = AUX.static_tasks(env, _res(), 0)
        self.assertFalse(done["dora"])
        bonus = AUX.bonus_of(done)
        # win + tsumo + no_dealin（该局没放铳） = 1000 + 300 + 200
        self.assertEqual(bonus, 1500)

    def test_dora_weight(self):
        env = type("E", (), {"players": [type("P", (), {"riichi": False})()
                                        for _ in range(4)]})()
        done = AUX.static_tasks(env, _res(dora=1), 0)
        self.assertEqual(AUX.bonus_of(done), 1500 + 300)
        self.assertEqual(AUX.bonus_of(done, {"dora": 2.0}), 1500 + 600)

    def test_game_result_carries_counts(self):
        r = _res(dora=2, aka=1, ura=3)
        self.assertEqual(r.dora_count, 2)
        self.assertEqual(r.aka_count, 1)
        self.assertEqual(r.ura_dora_count, 3)


if __name__ == "__main__":
    from tests.run_all import main  # type: ignore
    raise SystemExit(main())
