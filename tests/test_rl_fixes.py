"""RL 训练修复回归测试（纯 Python，不依赖 PyTorch）。

运行：python -m tests.test_rl_fixes
"""

from __future__ import annotations

import py_compile
import os
import unittest

import tests.utils as U
from mahjong import env as E
from mahjong import obs2 as O2
from mahjong import scenarios as SC
from mahjong import tiles as T


class TestDoraMapping(unittest.TestCase):
    def test_suit_and_honor_wraparound(self):
        # 9m -> 1m, 9p -> 1p, 9s -> 1s
        self.assertEqual(T.dora_from_indicator(8), 0)
        self.assertEqual(T.dora_from_indicator(17), 9)
        self.assertEqual(T.dora_from_indicator(26), 18)
        # 1m -> 2m, 5p -> 6p
        self.assertEqual(T.dora_from_indicator(0), 1)
        self.assertEqual(T.dora_from_indicator(13), 14)
        # 风牌：东(27)->南(28)->西(29)->北(30)->东(27)
        self.assertEqual(T.dora_from_indicator(27), 28)
        self.assertEqual(T.dora_from_indicator(30), 27)
        # 三元牌：白(31)->发(32)->中(33)->白(31)
        self.assertEqual(T.dora_from_indicator(31), 32)
        self.assertEqual(T.dora_from_indicator(33), 31)


class TestObsRelativeAndDora(unittest.TestCase):
    def test_relative_seat_rotation_and_dora_plane(self):
        env = U.make_env(seed=42)
        env.start_turn()
        env.dora_indicators = [8]  # 9m -> dora 1m (0)
        env.dora_indicator = 8
        env.players[0].points = 10000
        env.players[1].points = 20000
        env.players[2].points = 30000
        env.players[3].points = 40000
        env.players[2].river = [5]

        enc = O2.ObsEncoder()
        planes, scalar = enc.encode(env, 2)
        self.assertEqual(planes.shape, (O2.N_PLANES, 34))
        self.assertEqual(scalar.shape, (O2.SCALAR_DIM,))
        # 相对视角：玩家 2 的点数在 s[0]，顺序应为 [p2, p3, p0, p1]
        self.assertAlmostEqual(float(scalar[0]), 30000 / 25000.0, places=5)
        self.assertAlmostEqual(float(scalar[1]), 40000 / 25000.0, places=5)
        self.assertAlmostEqual(float(scalar[2]), 10000 / 25000.0, places=5)
        self.assertAlmostEqual(float(scalar[3]), 20000 / 25000.0, places=5)
        # 玩家 2 自己的牌河应在 planes[5]（相对 0 号位）
        self.assertEqual(float(planes[5, 5]), 1.0)
        # 9m 指示牌 -> 宝牌平面应在 1m (index 0)，而非 1p (index 9)
        self.assertEqual(float(planes[20, 0]), 1.0)
        self.assertEqual(float(planes[20, 9]), 0.0)


class TestRewardDefaults(unittest.TestCase):
    def test_no_double_dealin_or_free_survive_by_default(self):
        self.assertEqual(SC.W_DEAL_BASE, 0.0)
        self.assertEqual(SC.W_SURVIVE, 0.0)


class TestModifiedFilesCompile(unittest.TestCase):
    def test_all_modified_scripts_compile(self):
        files = [
            "mahjong/tiles.py",
            "mahjong/obs2.py",
            "mahjong/net.py",
            "mahjong/scenarios.py",
            "train_hanchan.py",
            "train_ppo.py",
            "train_reinforce.py",
            "train_dagger.py",
            "train_teacher_bc.py",
            "eval_hanchan.py",
        ]
        for rel in files:
            path = os.path.join(U.ROOT, rel)
            py_compile.compile(path, doraise=True)


if __name__ == "__main__":
    import sys
    sys.exit(U.run_module_tests(sys.modules[__name__]))
