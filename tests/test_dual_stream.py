"""双流 Transformer 框架与统一事件流回归测试。

运行：python -m tests.test_dual_stream
"""

from __future__ import annotations

import importlib.util
import os
import py_compile
import unittest

import numpy as np
import tests.utils as U
from mahjong import env as E
from mahjong import event_stream as ES
from mahjong import policies as POL

HAS_TORCH = importlib.util.find_spec("torch") is not None


class TestEventStream(unittest.TestCase):
    def test_bos_at_index_zero_on_fresh_env(self):
        env = U.make_env(seed=7)
        env.start_turn()
        ev = ES.encode_event_stream(env, p_idx=0, max_len=48)
        self.assertEqual(ev.shape, (48, 4))
        self.assertEqual(ev.dtype, np.int64)
        # 第 0 位为 BOS，确保全空开局时 Cross-Attn key_padding_mask 不全 True
        self.assertEqual(tuple(ev[0]), (1, 34, 0, 1))
        self.assertEqual(int(ev[1, 0]), 0)

    def test_captures_discard_tsumogiri_and_relative_seat(self):
        env = U.make_env(seed=11)
        env.start_turn()
        pols = {i: POL.make_policy("eff_wuke", seed=10 + i) for i in range(4)}
        for _ in range(12):
            if env.phase == E.Phase.END:
                break
            acts = {}
            for p in env.pending_players():
                legal = env.legal_actions(p)
                acts[p] = pols[p].choose(env, p, legal) if legal else E.PASS_ACTION
            res = env.step(acts)
            if res is not None:
                break

        ev = ES.encode_event_stream(env, p_idx=2, max_len=48)
        # 至少应捕获到弃牌事件（act_type in 1..6）
        non_bos = ev[1:]
        valid_rows = non_bos[non_bos[:, 0] > 0]
        self.assertGreater(len(valid_rows), 0)
        self.assertTrue((valid_rows[:, 0] >= 1).all() and (valid_rows[:, 0] <= 4).all())
        self.assertTrue((valid_rows[:, 1] >= 0).all() and (valid_rows[:, 1] <= 34).all())
        self.assertTrue((valid_rows[:, 2] >= 1).all() and (valid_rows[:, 2] <= 6).all())

    def test_dual_net_compiles(self):
        for rel in (
            "mahjong/event_stream.py",
            "mahjong/dual_net.py",
            "mahjong/env.py",
            "gen_teacher_data.py",
            "train_teacher_bc.py",
            "train_hanchan.py",
            "eval_hanchan.py",
        ):
            py_compile.compile(os.path.join(U.ROOT, rel), doraise=True)


@unittest.skipUnless(HAS_TORCH, "当前环境未安装 torch，跳过前向形状测试")
class TestDualStreamNetTorch(unittest.TestCase):
    def test_forward_and_aux_shapes_and_no_nan(self):
        import torch
        from mahjong.dual_net import DualStreamMahjongNet

        net = DualStreamMahjongNet(
            d_model=128, n_layers=3, n_heads=4,
            n_actions=47, touch_dim=8, extra_scalar=12,
        )
        net.eval()
        obs = torch.randn(4, 22 * 34 + 26 + 12)
        touch = torch.randn(4, 8)
        events = torch.zeros(4, 48, 4, dtype=torch.long)
        events[:, 0] = torch.tensor([1, 34, 0, 1])
        events[:, 1] = torch.tensor([2, 13, 2, 3])
        with torch.no_grad():
            logits, val, danger = net.forward_aux(obs, touch, events)
        self.assertEqual(tuple(logits.shape), (4, 47))
        self.assertEqual(tuple(val.shape), (4,))
        self.assertEqual(tuple(danger.shape), (4, 34))
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isnan(val).any())
        self.assertFalse(torch.isnan(danger).any())


if __name__ == "__main__":
    import sys
    sys.exit(U.run_module_tests(sys.modules[__name__]))
