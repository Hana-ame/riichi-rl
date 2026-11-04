"""实时性护栏测试（realtime_guard.py）。

运行：python -m tests.test_realtime_guard
"""

from __future__ import annotations

import time
import unittest

import tests.utils as U
from mahjong import env as E
from realtime_guard import LatencyStats, RealtimeGuard


class _FastAgent:
    name = "fast"

    def act(self, env, p, legal):
        return legal[0]

    def set_game(self, env, hstate=None):
        pass


class _SlowAgent(_FastAgent):
    name = "slow"

    def act(self, env, p, legal):
        time.sleep(0.02)          # 20ms，远超下面 1ms 预算
        return legal[0]


class TestLatencyStats(unittest.TestCase):
    def test_summary_percentiles(self):
        st = LatencyStats()
        for ms in (1, 2, 3, 4, 5):
            st.add(ms / 1000.0)
        s = st.summary()
        self.assertEqual(s["n"], 5)
        self.assertAlmostEqual(s["max_ms"], 5.0)
        self.assertAlmostEqual(s["p50_ms"], 3.0)

    def test_realtime_flag(self):
        st = LatencyStats()
        st.budget_ms = 5.0
        for ms in (1, 2, 2, 2):
            st.add(ms / 1000.0)
        self.assertTrue(st.summary()["realtime"])
        st.add(0.010)
        self.assertFalse(st.summary()["realtime"])


class TestRealtimeGuard(unittest.TestCase):
    def _env(self):
        env = U.make_env(seed=1)
        env.start_turn()
        return env

    def test_fast_agent_no_fallback(self):
        g = RealtimeGuard(_FastAgent(), budget_ms=5000)
        env = self._env()
        legal = env.legal_actions(0)
        a = g.act(env, 0, legal)
        self.assertIn(a, legal)
        self.assertEqual(g.stats.fallbacks, 0)
        self.assertTrue(g.stats.summary()["realtime"])

    def test_slow_agent_falls_back(self):
        g = RealtimeGuard(_SlowAgent(), budget_ms=1.0)
        env = self._env()
        legal = env.legal_actions(0)
        a = g.act(env, 0, legal)
        self.assertIn(a, legal)
        self.assertEqual(g.stats.fallbacks, 1, "超预算必须记一次 fallback")
        self.assertFalse(g.stats.summary()["realtime"])


if __name__ == "__main__":
    import sys
    sys.exit(U.run_module_tests(sys.modules[__name__]))
