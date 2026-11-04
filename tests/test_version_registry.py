"""版本登记表测试（version_registry.py）。

运行：python -m tests.test_version_registry
"""

from __future__ import annotations

import unittest

import torch

import tests.utils as U  # noqa: F401

from version_registry import arch_of, count_params, lineage_of


class TestLineageOf(unittest.TestCase):
    def test_families(self):
        cases = {
            "a2c_r10_it25.pt": "A2C（早期 RL）",
            "teacher47_v3.pt": "教师 BC 固定槽（teacher47）",
            "style_rl_v1.pt": "风格微调（style）",
            "from_games_v2.pt": "蒸馏（from_games）",
            "bc_v1_last.pt": "BC 人类牌谱教师",
            "riichi_net.pt": "纯 RL（riichi_net）",
            "hc_r13_it55.pt": "Hanchan HC",
        }
        for f, want in cases.items():
            self.assertEqual(lineage_of(f), want)

    def test_unknown(self):
        self.assertEqual(lineage_of("foo.pt"), "其他")


class TestArchOf(unittest.TestCase):
    def test_action_heads(self):
        self.assertEqual(arch_of({"policy.weight": torch.zeros(47, 4)}),
                         "hanchan47")
        self.assertEqual(arch_of({"policy.weight": torch.zeros(54, 4)}),
                         "convformer54")
        self.assertEqual(arch_of({"policy.weight": torch.zeros(243, 4)}),
                         "bc243")

    def test_predictors(self):
        self.assertEqual(arch_of({"trunk.0.weight": torch.zeros(1),
                                  "win.weight": torch.zeros(1)}),
                         "hand_predictor")
        self.assertEqual(arch_of({"net.0.weight": torch.zeros(1),
                                  "net.4.weight": torch.zeros(1)}),
                         "opponent_model")


class TestCountParams(unittest.TestCase):
    def test_sum(self):
        sd = {"a": torch.zeros(10), "b": torch.zeros(2, 3),
              "c": "not tensor"}
        self.assertEqual(count_params(sd), 16)


if __name__ == "__main__":
    import sys
    sys.exit(U.run_module_tests(sys.modules[__name__]))
