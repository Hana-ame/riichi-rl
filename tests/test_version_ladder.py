"""版本天梯晋级闸门测试（version_ladder.py）。

运行：python -m tests.test_version_ladder
"""

from __future__ import annotations

import unittest

import tests.utils as U  # noqa: F401

from version_ladder import ci95, verdict_of


class TestCILadder(unittest.TestCase):
    def test_ci_constant(self):
        m, lo, hi = ci95([100.0] * 30)
        self.assertAlmostEqual(m, 100.0)
        self.assertAlmostEqual(lo, 100.0)
        self.assertAlmostEqual(hi, 100.0)

    def test_ci_contains_mean(self):
        m, lo, hi = ci95([1.0, 2.0, 3.0, 4.0])
        self.assertTrue(lo <= m <= hi)


class TestVerdictGate(unittest.TestCase):
    def test_pass_when_clearly_stronger(self):
        v, _, lo, _ = verdict_of([500.0] * 40, elo_delta=0.0, band=50.0)
        self.assertEqual(v, "PASS")
        self.assertGreater(lo, 0)

    def test_pass_allowed_when_elo_within_noise(self):
        v, *_ = verdict_of([400.0] * 40, elo_delta=-30.0, band=50.0)
        self.assertEqual(v, "PASS", "Elo 小幅波动（在噪声带内）不阻断晋级")

    def test_fail_when_weaker(self):
        v, *_ = verdict_of([-300.0] * 40, elo_delta=0.0, band=50.0)
        self.assertEqual(v, "FAIL")

    def test_fail_when_elo_collapses_even_if_points_up(self):
        v, *_ = verdict_of([400.0] * 40, elo_delta=-200.0, band=50.0)
        self.assertEqual(v, "FAIL", "Elo 显著下滑不算晋级")

    def test_hold_when_ci_crosses_zero(self):
        diffs = [1000.0, -1000.0] * 20
        v, m, lo, hi = verdict_of(diffs, elo_delta=30.0, band=50.0)
        self.assertEqual(v, "HOLD")
        self.assertLess(lo, 0)
        self.assertGreater(hi, 0)

    def test_hold_when_not_enough_games(self):
        v, *_ = verdict_of([900.0] * 5, elo_delta=0.0, band=50.0)
        self.assertEqual(v, "HOLD", "样本不足不放过")


if __name__ == "__main__":
    import sys
    sys.exit(U.run_module_tests(sys.modules[__name__]))
