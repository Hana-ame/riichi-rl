"""牌河顺序特征 + 对手向听头测试（opp_shanten.py）。

钉死「顺序信息真的进了特征」：同一组牌河 counts，不同打出顺序 → 特征不同。
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from opp_shanten import Head, ORDER_LEN, order_planes, shanten_label


class TestRiverOrderFeatures(unittest.TestCase):
    def test_order_changes_features(self):
        # 同一组牌（3m×2 + 7m×1），两种不同打出顺序
        r1 = [[2, 2, 6], [], [], []]
        r2 = [[6, 2, 2], [], [], []]
        f1 = order_planes(r1)
        f2 = order_planes(r2)
        self.assertEqual(f1.shape, (4 * ORDER_LEN * 34,))
        self.assertFalse(np.array_equal(f1, f2), "顺序不同特征必须不同")

    def test_count_only_vectors_would_match_but_order_uses_position(self):
        # 位置平面：第 0 位是第 0 张、缺失位全 0
        r = [[2, 6], [], [], []]
        f = order_planes(r).reshape(4 * ORDER_LEN, 34)
        self.assertEqual(f[0, 2], 1.0)   # 第 0 位打 3m
        self.assertEqual(f[1, 6], 1.0)   # 第 1 位打 7m
        self.assertEqual(f[2].sum(), 0.0)  # 第 2 位为空

    def test_empty_rivers_are_zero(self):
        self.assertEqual(float(order_planes([[], [], [], []]).sum()), 0.0)


class TestShantenLabel(unittest.TestCase):
    def _counts(self, tiles):
        c = np.zeros(34, dtype=np.int8)
        for t in tiles:
            c[t] += 1
        return c

    def test_tenpai_label(self):
        sh, tp = shanten_label(self._counts(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 12, 13]))
        self.assertEqual(sh, 0)
        self.assertEqual(tp, 1.0)

    def test_noten_label(self):
        sh, tp = shanten_label(self._counts(
            [0, 1, 2, 3, 4, 9, 10, 18, 19, 27, 28, 29, 30]))
        self.assertGreater(sh, 0)
        self.assertEqual(tp, 0.0)


class TestHead(unittest.TestCase):
    def test_forward_shapes(self):
        m = Head(108)
        sh, tp = m(torch.zeros(3, 108))
        self.assertEqual(tuple(sh.shape), (3, 7))
        self.assertEqual(tuple(tp.shape), (3,))


if __name__ == "__main__":
    from tests.run_all import main  # type: ignore
    raise SystemExit(main())
