"""额外任务测试（mahjong/aux_tasks.py）。

运行：python -m tests.test_aux_tasks
"""

from __future__ import annotations

import unittest

import tests.utils as U
from mahjong import aux_tasks as AUX


def _forced_tsumo(env, winner=0):
    hand = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 27]
    U.set_hand(env, winner, hand)
    pl = env.players[winner]
    pl.last_drawn = 27
    pl.hand.append(27)
    pl.hand.sort()
    return env._finish_tsumo(winner)


class TestAuxStaticTasks(unittest.TestCase):
    def test_win_and_tsumo_tasks(self):
        env = U.make_env(seed=1, dealer=0)
        env.start_turn()
        r = _forced_tsumo(env, 0)
        done = AUX.static_tasks(env, r, 0)
        self.assertTrue(done["win"])
        self.assertTrue(done["tsumo"])
        self.assertTrue(done["no_dealin"])
        self.assertGreaterEqual(AUX.bonus_of(done),
                                AUX.TASKS["win"][0] + AUX.TASKS["tsumo"][0])

    def test_loser_gets_no_dealin_penalty(self):
        env = U.make_env(seed=2, dealer=0)
        env.start_turn()
        # 0 号听 5p，1 号放铳（用断幺荣和构造）
        U.set_hand(env, 0, [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16])
        env.last_discard = 13
        env.last_discard_player = 1
        env.phase = U.E.Phase.RESPONSE
        r = env._finish_ron(0, 1, 13)
        self.assertEqual(r.result_type, "ron")
        win_done = AUX.static_tasks(env, r, 0)
        lose_done = AUX.static_tasks(env, r, 1)
        self.assertTrue(win_done["no_dealin"])
        self.assertTrue(win_done["win"])
        self.assertFalse(lose_done["no_dealin"], "放铳者不算完成未放铳任务")

    def test_riichi_task(self):
        env = U.make_env(seed=3)
        env.start_turn()
        env.players[2].riichi = True
        r = env._finish_ryuukyoku(reason="exhaustive")
        done = AUX.static_tasks(env, r, 2)
        self.assertTrue(done["riichi"])

    def test_tenpai_draw_task(self):
        env = U.make_env(seed=4)
        env.start_turn()
        # 0 号听牌（13 张），其余留 14/13 任意；流局时 0 号应完成「流局听牌」
        U.set_hand(env, 0, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 12, 13])
        r = env._finish_ryuukyoku(reason="exhaustive")
        self.assertEqual(r.result_type, "ryuukyoku")
        done = AUX.static_tasks(env, r, 0)
        self.assertTrue(done["tenpai_draw"])
        self.assertTrue(done["no_dealin"])


class TestAuxTracker(unittest.TestCase):
    def test_fast_tenpai_tracked(self):
        env = U.make_env(seed=5)
        env.start_turn()
        env.turn = 3                     # 8 巡内
        U.set_hand(env, 0, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 12, 13])
        tr = AUX.AuxTaskTracker(0)
        tr.observe(env)
        self.assertTrue(tr.fast_tenpai)
        done, bonus = tr.finish(env, None)
        self.assertTrue(done["fast_tenpai"])
        self.assertGreaterEqual(bonus, AUX.TASKS["fast_tenpai"][0])

    def test_tracker_ready_only_bonus_after_deal(self):
        env = U.make_env(seed=6)
        env.start_turn()
        tr = AUX.AuxTaskTracker(1)
        tr.observe(env)                  # 还没进听
        done, _ = tr.finish(env, None)
        self.assertFalse(done["fast_tenpai"])


if __name__ == "__main__":
    import sys
    sys.exit(U.run_module_tests(sys.modules[__name__]))