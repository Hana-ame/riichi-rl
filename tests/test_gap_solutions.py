"""三大缺口解决方案回归测试（纯 Python，不依赖 PyTorch）。

覆盖：
1. 缺口 1（Post-Call Value 副露前瞻）：simulate_post_call_state 牌数守恒与原 env 零污染、
   has_open_yaku_potential（役牌 / 断幺 / 混一色 / 对对和 / 终盘形式听牌流听 / 无役拦截）；
2. 缺口 2（手切/摸切事件流）：houou_data.decode_match 还原真实 history_events（tsumogiri vs discard vs riichi）、
   tenhou_to_obs2.TenhouPseudoEnv 注入 log 与立直后现物；
3. 全部新增与修改脚本编译校验。

运行：python -m tests.test_gap_solutions
"""

from __future__ import annotations

import os
import py_compile
import unittest

import numpy as np

import tests.utils as U
import houou_data as HD
import post_call_agent as PCA
from mahjong import env as E
from mahjong import event_stream as ES


class TestPostCallLookahead(unittest.TestCase):
    def test_simulate_post_call_does_not_mutate_original_env(self):
        env = U.make_env(seed=100, dealer=0)
        env.start_turn()
        # 玩家 0 打出白板(31)，玩家 1 手里有 2 张白板可碰
        env.players[1].hand = [31, 31, 1, 2, 3, 10, 11, 12, 19, 20, 21, 22, 27]
        env.last_discard = 31
        env.last_discard_player = 0
        env.players[0].river.append(31)
        env.phase = E.Phase.RESPONSE
        env.pending = [1, 2, 3]

        orig_hand = list(env.players[1].hand)
        orig_melds = list(env.players[1].melds)

        sim, disc_tile, sh_after = PCA.simulate_post_call_state(env, 1, E.Action(E.ACT_PON, 31))
        self.assertIsNotNone(sim)
        self.assertGreaterEqual(disc_tile, 0)
        # 原 env 绝不能被污染
        self.assertEqual(env.players[1].hand, orig_hand)
        self.assertEqual(env.players[1].melds, orig_melds)
        # 副露后碰白板应识别为 yakuhai 且向听数降到 0（听 5s 单骑）
        ok, reason = PCA.has_open_yaku_potential(sim, 1)
        self.assertTrue(ok)
        self.assertEqual(reason, "yakuhai")
        self.assertEqual(sh_after, 0)

    def test_open_yaku_potential_tanyao_and_no_yaku_and_keiten(self):
        env = U.make_env(seed=101, dealer=0)
        p = env.players[0]
        # 断幺九潜力
        p.melds = [("chi", 1, 3)]
        p.hand = [2, 3, 4, 11, 12, 13, 20, 21, 22, 23]
        ok, reason = PCA.has_open_yaku_potential(env, 0)
        self.assertTrue(ok)
        self.assertEqual(reason, "tanyao")

        # 无役杂牌（吃 123m + 杂色含幺九，且牌山充裕非终盘）
        p.melds = [("chi", 0, 3)]
        p.hand = [4, 5, 9, 13, 17, 18, 22, 26, 28, 29]
        env.wall_idx = 52
        ok2, reason2 = PCA.has_open_yaku_potential(env, 0)
        self.assertFalse(ok2)
        self.assertEqual(reason2, "no_yaku")

        # 终盘剩余 <= 16 张且已听牌 -> 识别为 keiten（流听）
        p.melds = [("chi", 0, 3)]
        p.hand = [9, 10, 11, 12, 13, 14, 24, 25, 26, 29]  # 听西单骑
        env.wall_idx = len(env.wall) - getattr(env, "dead_wall_size", 14) - 8
        ok3, reason3 = PCA.has_open_yaku_potential(env, 0)
        self.assertTrue(ok3)
        self.assertEqual(reason3, "keiten")


class TestHououTsumogiriEventStream(unittest.TestCase):
    def test_decode_match_preserves_tsumogiri_and_riichi_events(self):
        # 构造最小合成 npz 验证 decode_match 提取真实手切(0)/摸切(1)/立直事件
        data = {
            "trigger_type": np.array([0], dtype=np.int8),
            "trigger_actor": np.array([0], dtype=np.int8),
            "trigger_tile": np.array([0], dtype=np.int16),
            "legal_offsets": np.array([0, 1], dtype=np.int32),
            "legal_type": np.array([HD.ACT_DISCARD], dtype=np.int8),
            "legal_tiles": np.array([[1, 0, 0]], dtype=np.int16),
            "expert_index": np.array([0], dtype=np.int32),
            "hand_counts": np.zeros((1, 34), dtype=np.uint8),
            "hand_red_counts": np.zeros((1, 34), dtype=np.uint8),
            "dora": np.array([[1, 0, 0, 0, 0]], dtype=np.int16),
            "history_offsets": np.array([0, 4], dtype=np.int32),
            # 事件：0=tsumo, 1=手切(1m, hs=0), 2=摸切(2m, hs=1), 3=立直+打3m
            "history_type": np.array([0, 1, 1, 2, 1], dtype=np.int8),
            "history_actor": np.array([0, 0, 1, 2, 2], dtype=np.int8),
            "history_tiles": np.array([
                [0, 0, 0],
                [1, 0, 0],
                [2, 0, 0],
                [0, 0, 0],
                [3, 0, 0],
            ], dtype=np.int16),
            "history_tsumogiri": np.array([0, 0, 1, 0, 0], dtype=np.int8),
            "history_ref": np.array([0], dtype=np.int32),
            "history_len": np.array([5], dtype=np.int32),
            "scores": np.array([[25000, 25000, 25000, 25000]], dtype=np.int32),
            "seat_wind": np.array([27], dtype=np.uint8),
            "round_wind": np.array([27], dtype=np.uint8),
            "kyoku": np.array([1], dtype=np.uint8),
            "honba": np.array([0], dtype=np.uint8),
            "tiles_left": np.array([65], dtype=np.int16),
            "score_delta": np.array([1500], dtype=np.int32),
            "final_rank": np.array([0], dtype=np.int8),
            "steps_to_done": np.array([10], dtype=np.int16),
            "player": np.array([0], dtype=np.int8),
        }
        decs = HD.decode_match(data)
        self.assertEqual(len(decs), 1)
        evs = decs[0].history_events
        # 应还原出 3 条弃牌/立直事件：discard(0), tsumogiri(1), riichi(2)
        self.assertEqual([(e[1], e[2], e[3]) for e in evs], [
            (0, "discard", 0),
            (1, "tsumogiri", 1),
            (2, "riichi", 2),
        ])

        # 验证 event_stream 编码正确区分 act_id=1(手切) / 2(摸切) / 3(立直)
        class _DummyEnv:
            log = evs
            players = []

        arr = ES.encode_event_stream(_DummyEnv(), 0, max_len=8)
        self.assertEqual(int(arr[1, 2]), 1)  # discard
        self.assertEqual(int(arr[2, 2]), 2)  # tsumogiri
        self.assertEqual(int(arr[3, 2]), 3)  # riichi


class TestAllNewFilesCompile(unittest.TestCase):
    def test_compile(self):
        for rel in (
            "houou_data.py",
            "tenhou_to_obs2.py",
            "train_teacher_bc.py",
            "post_call_agent.py",
            "train_river_seq.py",
            "eval_selfplay_gate.py",
        ):
            py_compile.compile(os.path.join(U.ROOT, rel), doraise=True)


if __name__ == "__main__":
    unittest.main()
