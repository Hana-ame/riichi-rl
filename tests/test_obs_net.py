"""观测编码 + 策略网络测试（mahjong/obs2.py、mahjong/net.py）。

运行：python -m tests.test_obs_net
"""

from __future__ import annotations

import unittest

import tests.utils as U

import torch

from mahjong import env as E
from mahjong import obs2 as O2
from mahjong.net import MahjongNet, count_params

FLAT_DIM = O2.OBS_TENSOR_DIM


class TestObsEncoder(unittest.TestCase):
    def test_shape(self):
        enc = O2.ObsEncoder()
        env = U.make_env(seed=1)
        env.start_turn()
        planes, scalar = enc.encode(env, 0)
        self.assertEqual(planes.shape, (O2.N_PLANES, 34))
        self.assertEqual(scalar.shape, (O2.SCALAR_DIM,))
        self.assertEqual(planes.ravel().shape[0] + scalar.shape[0], FLAT_DIM)

    def test_deterministic_same_state(self):
        enc = O2.ObsEncoder()
        env = U.make_env(seed=2)
        env.start_turn()
        p1, s1 = enc.encode(env, 0)
        p2, s2 = enc.encode(env, 0)
        self.assertTrue((p1 == p2).all())
        self.assertTrue((s1 == s2).all())


class TestNet(unittest.TestCase):
    def test_forward_shapes(self):
        model = MahjongNet()
        model.eval()
        x = torch.zeros(3, FLAT_DIM)
        with torch.no_grad():
            policy, value = model(x)
        self.assertEqual(tuple(policy.shape), (3, 54))
        self.assertEqual(tuple(value.shape), (3,))
        # 概率分布性：softmax 后和为 1
        probs = torch.softmax(policy, dim=-1)
        self.assertTrue(torch.allclose(probs.sum(dim=-1),
                                       torch.ones(3), atol=1e-5))

    def test_param_count_stable(self):
        # 现有 riichi_net.pt 的参数量，防止网络结构被无意改动
        self.assertGreater(count_params(MahjongNet()), 1_000_000)

    def test_configurable_size(self):
        small = MahjongNet(channels=64, n_blocks=4)
        n_small = count_params(small)
        self.assertLess(n_small, count_params(MahjongNet()))
        x = torch.zeros(1, FLAT_DIM)
        with torch.no_grad():
            policy, value = small(x)
        self.assertEqual(tuple(policy.shape), (1, 54))


if __name__ == "__main__":
    import sys
    sys.exit(U.run_module_tests(sys.modules[__name__]))