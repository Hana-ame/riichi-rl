"""预测模型测试：对手手牌推断 / 和牌预测器。

覆盖：opponent_model.py、hand_predictor.py 的特征维度和前向输出形状/值域。

运行：python -m tests.test_models
"""

from __future__ import annotations

import glob
import os
import unittest

import tests.utils as U

import numpy as np
import torch

from hand_predictor import IN_DIM as HP_IN, HandPredictor, build_feature as hp_build
from opponent_model import IN_DIM as OPP_IN, OpponentNet, build_features


class TestOpponentModel(unittest.TestCase):
    def test_feature_dim(self):
        hand = np.zeros(34, np.int8); hand[0] = 2
        river = np.zeros(34, np.int8); river[1] = 1
        other = np.zeros(34, np.int8); other[2] = 3
        f = build_features(hand, river, other, target_melds=1, target_riichi=0,
                           target_discards=5, left=60, dora=10)
        self.assertEqual(f.shape, (OPP_IN,))
        self.assertEqual(OPP_IN, 34 * 3 + 6)

    def test_forward(self):
        m = OpponentNet()
        m.eval()
        x = torch.randn(4, OPP_IN)
        with torch.no_grad():
            y = m(x)
        self.assertEqual(tuple(y.shape), (4, 34))
        self.assertTrue(float(y.min()) >= 0.0)
        self.assertTrue(float(y.max()) <= 1.0)


class TestHandPredictor(unittest.TestCase):
    def test_feature_dim(self):
        hand = np.zeros(34, np.int8)
        f = hp_build(hand, melds=0, riichi=0, shanten=2, left=70, dora=5)
        self.assertEqual(f.shape, (HP_IN,))
        self.assertEqual(HP_IN, 34 + 6)

    def test_forward(self):
        m = HandPredictor()
        m.eval()
        x = torch.randn(3, HP_IN)
        with torch.no_grad():
            pw, pp, pt = m(x)
        self.assertEqual(tuple(pw.shape), (3, 1))
        self.assertEqual(tuple(pp.shape), (3,))
        self.assertEqual(tuple(pt.shape), (3,))
        self.assertTrue(float(pw.min()) >= 0.0 and float(pw.max()) <= 1.0)
        self.assertTrue(float(pt.min()) >= 0.0, "听牌巡数应非负")


class TestHandPredictorData(unittest.TestCase):
    """回归：和牌预测器读流局记录时 winner=None 不能炸（2026-11-05 修复）。

    旧 `load_dataset` 直接 `int(d["winner"])`，703 局里有 97 局是流局
    （winner=None）→ TypeError，`hand_predictor.py` 训练必 exit 1。
    """

    def test_winner_index_none_means_ryuukyoku(self):
        from hand_predictor import winner_index
        import numpy as np
        self.assertEqual(winner_index({"winner": np.array(None, dtype=object)}), -1)
        self.assertEqual(winner_index({"winner": np.array(2)}), 2)
        self.assertEqual(winner_index({"winner": 3}), 3)


_HANCHAN_CKS = sorted(glob.glob(os.path.join(
    U.ROOT, "checkpoints", "a2c_r*_it*.pt")))


@unittest.skipUnless(_HANCHAN_CKS,
                     "checkpoints/a2c_r*_it*.pt 不存在（磁盘权重不入库）")
class TestEvalHanchanLoading(unittest.TestCase):
    """回归：eval_hanchan 必须能加载 47 槽半庄 ckpt（2026-11-05 修复）。

    旧版 `load_model` 固定按 54 槽 MahjongNet 建网，而默认 glob 选中的
    `a2c_r*_it*.pt` 是 47 槽半庄模型 → `load_state_dict` size mismatch，
    `eval_hanchan.py` 开箱即崩。
    """

    def test_load_47_slot_ckpt(self):
        from train_hanchan import N_ACTIONS
        from eval_hanchan import load_model
        model, ck = load_model(_HANCHAN_CKS[0])
        self.assertIsNotNone(ck.get("iter"))
        self.assertEqual(model.policy.out_features, N_ACTIONS,
                         "半庄线 ckpt 必须按 47 槽建网加载")


if __name__ == "__main__":
    import sys
    sys.exit(U.run_module_tests(sys.modules[__name__]))