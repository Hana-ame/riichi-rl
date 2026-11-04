"""半庄场次测试（mahjong/hanchan.py）——点数接力与供托。

运行：python -m tests.test_hanchan
"""

from __future__ import annotations

import unittest

import tests.utils as U
from mahjong import env as E
from mahjong.hanchan import HanchanState


def _forced_tsumo(env, winner=0, hand=None):
    """构造必和牌型并结算（沿用 test_env_core 的构造手法）。"""
    if hand is None:
        hand = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 27]
    U.set_hand(env, winner, hand)
    env.players[winner].last_drawn = 27
    env.players[winner].hand.append(27)
    env.players[winner].hand.sort()
    env.turn_player = winner
    return env._finish_tsumo(winner)


class TestStartPointsCarry(unittest.TestCase):
    """回归：上一局结算后的点数必须成为下一局起手（持越点）。

    修复前每局都从 25000 重开：`HanchanState.sticks` 只在外面记流水，
    局内立直资格（<1000 不能立直）与观测里的点数特征全部失真。
    """

    def test_env_honors_start_points(self):
        env = E.MahjongEnv(seed=1, start_points=[24000, 25000, 26000, 25000])
        self.assertEqual([p.points for p in env.players],
                         [24000, 25000, 26000, 25000])

    def test_settle_writes_back_points(self):
        before = [24000, 25000, 26000, 25000]
        env = E.MahjongEnv(seed=1, start_points=before)
        r = _forced_tsumo(env, 0)
        self.assertEqual([p.points for p in env.players], r.scores,
                         "结算后 env 内点数必须等于 result.scores")
        self.assertEqual(sum(r.scores) + r.kyokuten, sum(before),
                         "点数 + 桌上供托必须守恒")

    def test_new_env_uses_sticks(self):
        h = HanchanState(dealer=0)
        h.sticks = [30000, 20000, 26000, 24000]
        env = h.new_env(seed=7)
        self.assertEqual([p.points for p in env.players], h.sticks)

    def test_advance_carries_scores_and_kyokuten(self):
        h = HanchanState(dealer=0)
        h.sticks = [24000, 25000, 27000, 24000]
        env = h.new_env(seed=3)
        r = _forced_tsumo(env, 0)
        h.advance(r)
        self.assertEqual(sum(h.sticks) + h.kyokuten, 100000,
                         "半庄内「sticks 总和 + 供托」必须恒为 100000")
        env2 = h.new_env(seed=4)
        self.assertEqual([p.points for p in env2.players], h.sticks)

    def test_advance_handles_old_style_result(self):
        """兼容只有 points_delta 的旧式/测试 result（无 scores 字段）。"""
        class _R:
            result_type = "ryuukyoku"
            winner = None
            points_delta = [0, -1000, 1000, 0]
            kyokuten = 0

        h = HanchanState(dealer=0)
        h.advance(_R())          # 不应抛异常
        self.assertEqual(sum(h.sticks), 100000)


if __name__ == "__main__":
    unittest.main()