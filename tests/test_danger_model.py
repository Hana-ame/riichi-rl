"""对手「和牌张」危险模型测试（danger_model.py）。

运行：python -m tests.test_danger_model
"""

from __future__ import annotations

import os
import unittest

import numpy as np
import torch

import tests.utils as U
from danger_model import DangerNet, waits_of


class TestWaitsOf(unittest.TestCase):
    def _counts(self, tiles):
        c = np.zeros(34, dtype=np.float32)
        for t in tiles:
            c[t] += 1
        return c

    def test_tenpai_hand_has_waits(self):
        # 123m456m789m 11p 45p：听 3p(11)/6p(14)
        y, tenpai = waits_of(self._counts(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 12, 13]))
        self.assertTrue(tenpai)
        self.assertGreater(float(y.sum()), 0)
        self.assertEqual(float(y[14]), 1.0)

    def test_not_tenpai_gives_zero(self):
        y, tenpai = waits_of(self._counts([0, 1, 2, 3, 4, 9, 10, 18, 19, 27,
                                          28, 29, 30]))
        self.assertFalse(tenpai)
        self.assertEqual(float(y.sum()), 0.0)

    def test_fourteen_tiles_rejected(self):
        y, tenpai = waits_of(np.ones(34, dtype=np.float32) * 0.1)
        self.assertFalse(tenpai)


class TestDangerNet(unittest.TestCase):
    @unittest.skipUnless(os.path.exists("checkpoints/danger_model.pt"),
                         "无 danger_model.pt（磁盘权重不入库）")
    def test_forward_shape_and_range(self):
        ck = torch.load("checkpoints/danger_model.pt", map_location="cpu",
                        weights_only=False)
        m = DangerNet()
        m.load_state_dict(ck["state_dict"])
        m.eval()
        with torch.no_grad():
            p = m(torch.zeros(2, 108))
        self.assertEqual(tuple(p.shape), (2, 34))
        self.assertTrue(float(p.min()) >= 0.0 and float(p.max()) <= 1.0)


if __name__ == "__main__":
    import sys
    sys.exit(U.run_module_tests(sys.modules[__name__]))
