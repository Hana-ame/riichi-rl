"""竞技场计分测试（arena_elo.py）——Elo 更新必须零和。

运行：python -m tests.test_arena
"""

from __future__ import annotations

import unittest

import tests.utils as U  # noqa: F401  (确保 ROOT 在 sys.path)

from arena_elo import elo_update, render_markdown


class TestEloZeroSum(unittest.TestCase):
    """回归：逐对 Elo 更新必须零和，否则全体评分会整体漂移。"""

    def test_pairwise_update_zero_sum(self):
        ratings = {0: 1500.0, 1: 1500.0, 2: 1500.0, 3: 1500.0}
        sel = [0, 1, 2, 3]
        before = sum(ratings.values())
        # 名次 0>1>2>3（点数降序）
        elo_update(ratings, sel, deltas=[30000.0, 20000.0, 10000.0, 0.0])
        self.assertAlmostEqual(sum(ratings.values()), before, places=6,
                               msg="一局 4 人 Elo 变化之和必须为 0")

    def test_converges_under_dominant_winner(self):
        """同一名次结果重复时，强者评分应上升、弱者下降。"""
        ratings = {0: 1500.0, 1: 1500.0, 2: 1500.0, 3: 1500.0}
        order = [1000.0, 800.0, 600.0, 400.0]
        for _ in range(50):
            elo_update(ratings, [0, 1, 2, 3], order)
        self.assertGreater(ratings[0], ratings[3])
        self.assertAlmostEqual(sum(ratings.values()),
                               sum([1500.0] * 4), places=6)


class TestEloMarkdown(unittest.TestCase):
    """ELO.md 渲染：局数/判读带宽必须来自数据，而不是写死。"""

    def test_render_includes_dynamic_header(self):
        rows = [
            {"name": "A", "kind": "rule", "elo": 1600.0,
             "games": 300, "win_rate": 10.0, "avg_pt": 100.0},
            {"name": "B", "kind": "v1_54", "elo": 1400.0,
             "games": 300, "win_rate": 0.0, "avg_pt": -100.0},
        ]
        md = render_markdown(rows, 600)
        self.assertIn("> 600 局", md)
        self.assertIn("每人 300–300 局", md)
        self.assertIn("| 1 | A |", md)
        self.assertIn("| 2 | B |", md)


if __name__ == "__main__":
    unittest.main()