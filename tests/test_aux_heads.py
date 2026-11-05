"""Stage 1 辅助头测试（aux_heads.py）。"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from aux_heads import AuxHeads, _auc


class TestAuxHeads(unittest.TestCase):
    def test_forward_shapes(self):
        m = AuxHeads(108)
        sh, tp, win = m(torch.zeros(4, 108))
        self.assertEqual(tuple(sh.shape), (4, 7))
        self.assertEqual(tuple(tp.shape), (4,))
        self.assertEqual(tuple(win.shape), (4,))

    def test_auc_perfect(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0.0, 0.0, 1.0, 1.0])
        self.assertAlmostEqual(_auc(scores, labels), 1.0, places=6)

    def test_auc_random_like(self):
        scores = np.array([0.4, 0.6, 0.5, 0.7])
        labels = np.array([0.0, 1.0, 0.0, 1.0])
        auc = _auc(scores, labels)
        self.assertTrue(0.0 <= auc <= 1.0)


if __name__ == "__main__":
    from tests.run_all import main  # type: ignore
    raise SystemExit(main())
