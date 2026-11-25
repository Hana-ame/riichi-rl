"""数据管线测试：动作编码桥接 + 数据生成序号。

覆盖：action_bridge.py、train_reinforce.slot_of、gen_greedy_data.next_seq

运行：python -m tests.test_data_pipeline
"""

from __future__ import annotations

import os
import tempfile
import unittest

import tests.utils as U

import action_bridge as AB
from mahjong import env as E
from train_hanchan import N_ACTIONS, slot_of


class TestSlotOf(unittest.TestCase):
    def test_discard_maps_to_tile(self):
        for tile in range(34):
            a = E.Action(E.ACT_DISCARD, tile)
            self.assertEqual(slot_of(a), tile, "打牌动作必须映射到 0..33")

    def test_special_actions_in_special_range(self):
        specials = [
            E.Action(E.ACT_TSUMO),
            E.Action(E.ACT_RIICHI, 0),
            E.Action(E.ACT_ANKAN, 0),
            E.Action(E.ACT_CHI, 0, 1),
            E.Action(E.ACT_PON, 5),
            E.Action(E.ACT_RON, 7),
            E.PASS_ACTION,
        ]
        for a in specials:
            s = slot_of(a)
            self.assertGreaterEqual(s, 34)
            self.assertLess(s, N_ACTIONS)

    def test_all_slots_in_range(self):
        for t in range(12):
            for tile in list(range(34)) + [None]:
                try:
                    a = E.Action(t, tile)
                except Exception:
                    continue
                s = slot_of(a)
                if s is None:
                    continue
                self.assertGreaterEqual(s, 0)
                self.assertLess(s, N_ACTIONS)


class TestActionBridge(unittest.TestCase):
    def test_mapping_tables_consistent(self):
        for ds, env_t in AB.DS_TO_ENV.items():
            self.assertEqual(AB.ENV_TO_DS[env_t], ds,
                             f"数据集动作 {ds} ↔ 引擎动作 {env_t} 双向映射必须一致")

    def test_env_action_to_slot_discard(self):
        # 打牌走 TILED_SLOT_BASE[ds=0] + tile
        a = E.Action(E.ACT_DISCARD, 5)
        s = AB.env_action_to_slot(a, {0: 0}, {}, {0: 1})
        self.assertEqual(s, 5)

    def test_unknown_type_returns_minus_one(self):
        a = E.Action(99, 0)
        self.assertEqual(AB.env_action_to_slot(a, {}, {}, {}), -1)


class TestGenSeq(unittest.TestCase):
    def test_next_seq_resumes(self):
        import gen_greedy_data as G
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(G.next_seq(d), 1)
            # 创建 g00007.npz 后应从 8 开始（防止重启覆盖旧数据）
            open(os.path.join(d, "g00007.npz"), "wb").close()
            self.assertEqual(G.next_seq(d), 8)


if __name__ == "__main__":
    import sys
    sys.exit(U.run_module_tests(sys.modules[__name__]))