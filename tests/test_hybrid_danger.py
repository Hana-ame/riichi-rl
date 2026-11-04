"""混合危险张 Agent 测试（hybrid_agents.HybridShantenDangerAgent）。

运行：python -m tests.test_hybrid_danger
"""

from __future__ import annotations

import unittest

import numpy as np

import tests.utils as U
from mahjong import env as E
from hybrid_agents import HybridShantenDangerAgent


class _NoModel(HybridShantenDangerAgent):
    """不加载权重的最小实例：只测 danger_scores 的纯逻辑。"""

    def __init__(self):
        self.opp = None
        self.n_actions = 47
        self.is47 = True
        self.mode = "hanchan"


class TestDangerScores(unittest.TestCase):
    def test_aggregates_probs_by_threat_weight(self):
        a = _NoModel()
        env = U.make_env(seed=3)
        env.start_turn()
        # 对手 1 立直、对手 2 未立直
        env.players[1].riichi = True
        probs = {
            1: np.eye(34)[4],        # 100% 持 5m(4)
            2: np.eye(34)[13] * 0.5,  # 50% 持 5p(13)
        }
        a._opp_hold_probs = lambda env, p: probs
        danger = a.danger_scores(env, 0)
        self.assertAlmostEqual(float(danger[4]), 1.0, places=5)
        self.assertAlmostEqual(float(danger[13]), 0.125, places=5)
        self.assertEqual(float(danger.sum()), 1.125)

    def test_no_model_returns_zeros(self):
        a = _NoModel()
        env = U.make_env(seed=4)
        env.start_turn()
        danger = a.danger_scores(env, 0)
        self.assertEqual(float(danger.sum()), 0.0)


if __name__ == "__main__":
    import sys
    sys.exit(U.run_module_tests(sys.modules[__name__]))
