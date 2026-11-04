"""牌编码模块测试（mahjong/tiles.py）。

运行：python -m tests.test_tiles
"""

from __future__ import annotations

import unittest

import tests.utils as U  # noqa: F401

from mahjong.tiles import NUM_TILES, NUM_TOTAL_TILES, counts_str, is_honor, tile_name


class TestTiles(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(NUM_TILES, 34)
        self.assertEqual(NUM_TOTAL_TILES, 136)

    def test_tile_name_roundtrip(self):
        names = [tile_name(t) for t in range(34)]
        self.assertEqual(len(names), 34)
        self.assertEqual(len(set(names)), 34, "34 种牌的显示名必须互不相同")
        for n in names:
            self.assertTrue(n and isinstance(n, str))

    def test_honor_range(self):
        for t in range(27):
            self.assertFalse(is_honor(t), f"{tile_name(t)} 不应是字牌")
        for t in range(27, 34):
            self.assertTrue(is_honor(t), f"{tile_name(t)} 应是字牌")

    def test_counts_str(self):
        c = [0] * 34
        c[0] = c[1] = c[2] = 1   # 123m
        c[9] = 1                 # 1p
        s = counts_str(c)
        self.assertIn("123", s)
        self.assertIn("m", s)
        self.assertIn("p", s)

    def test_counts_str_empty(self):
        self.assertEqual(counts_str([0] * 34), "")


if __name__ == "__main__":
    import sys
    sys.exit(U.run_module_tests(sys.modules[__name__]))