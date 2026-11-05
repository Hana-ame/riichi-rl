"""证据测试：我们的对手模型/和牌预测器**没有**牌河顺序（order）信息。

审阅要求「通过牌河顺序判断他家形状/向听」。当前实现：
- opponent_model.build_features 的牌河入参是 **每张牌计数（34 维 counts）**，
  不是弃牌序列；同一组计数无论按什么顺序打出，特征完全一致；
- hand_predictor.build_feature 根本不接受牌河（只有自家手牌 + 标量）。
本测试把「顺序无关」钉死成回归测试，防止以后误以为已有顺序特征。
"""

from __future__ import annotations

import inspect
import unittest

import numpy as np

from hand_predictor import build_feature as hp_build
from opponent_model import N_TILES, build_features


class TestNoRiverOrder(unittest.TestCase):
    def test_opponent_features_ignore_river_order(self):
        # 同一组弃牌计数（3m×2 + 7m×1），两种"顺序"必然对应同一个 counts 向量
        river_a = np.zeros(N_TILES, dtype=np.int8)
        river_a[2] = 2
        river_a[6] = 1
        river_b = np.zeros(N_TILES, dtype=np.int8)
        river_b[6] = 1
        river_b[2] = 2      # 同一 counts，仅赋值顺序不同
        hand = np.zeros(N_TILES, dtype=np.int8)
        other = np.zeros(N_TILES, dtype=np.int8)
        fa = build_features(hand, river_a, other, 0, 0, 3, 70, 4)
        fb = build_features(hand, river_b, other, 0, 0, 3, 70, 4)
        self.assertEqual(fa.shape, fb.shape)
        np.testing.assert_array_equal(fa, fb)

    def test_opponent_build_features_has_no_order_parameter(self):
        params = list(inspect.signature(build_features).parameters)
        # 牌河参数是 counts；不存在 discard order/history 参数
        self.assertIn("target_river", params)
        for bad in ("history", "order", "discard_seq", "sequence"):
            self.assertNotIn(bad, params)

    def test_hand_predictor_takes_no_river_at_all(self):
        params = list(inspect.signature(hp_build).parameters)
        self.assertEqual(params, ["hand_counts", "melds", "riichi",
                                  "shanten", "left", "dora"])
        self.assertFalse(any("river" in p for p in params))


if __name__ == "__main__":
    from tests.run_all import main  # type: ignore
    raise SystemExit(main())
