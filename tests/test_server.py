"""服务/记录层测试：观战服务器记录结构 + LLM 玩家动作解析。

覆盖：serve_game.py（player_json/remain_counts/state_json）、llm_player.py（parse_choice/action_label）

运行：python -m tests.test_server
"""

from __future__ import annotations

import unittest

import tests.utils as U

from mahjong import env as E
import llm_player as LP
import serve_game as SG


class TestLLMParsing(unittest.TestCase):
    def test_parse_choice_valid(self):
        self.assertEqual(LP.parse_choice("3", 5), 3)
        self.assertEqual(LP.parse_choice(" 选 2 ", 5), 2)
        self.assertEqual(LP.parse_choice("编号：4", 5), 4)

    def test_parse_choice_invalid(self):
        self.assertIsNone(LP.parse_choice("不知道", 5))
        self.assertIsNone(LP.parse_choice("9", 5), "越界编号必须返回 None")
        self.assertIsNone(LP.parse_choice("0", 5))

    def test_action_label(self):
        disc = E.Action(E.ACT_DISCARD, 0)
        label = LP.action_label(disc)
        self.assertIn("打", label)
        pass_label = LP.action_label(E.PASS_ACTION)
        self.assertIn("跳过", pass_label)


class TestRecordStructures(unittest.TestCase):
    def test_player_json(self):
        env = U.make_env(seed=1)
        d = SG.player_json(env.players[0])
        for key in ("hand", "river", "melds", "riichi", "points", "riichi_turn"):
            self.assertIn(key, d)
        self.assertEqual(len(d["hand"]), 13)

    def test_remain_counts(self):
        env = U.make_env(seed=2)
        r = SG.remain_counts(env)
        self.assertEqual(len(r), 34)
        for v in r:
            self.assertGreaterEqual(v, 0)
            self.assertLessEqual(v, 4)

    def test_state_json(self):
        env = U.make_env(seed=3)
        env.start_turn()
        d = SG.state_json(env, turn=0)
        self.assertEqual(d["type"], "state")
        self.assertEqual(len(d["players"]), 4)
        self.assertEqual(len(d["per_seat"]), 4)
        self.assertEqual(len(d["stats"]), 4)
        self.assertEqual(len(d["tsumo_last"]), 4)
        self.assertIn("opred", d)
        self.assertIn("hpred", d)
        import json
        json.dumps(d)   # 必须可 JSON 序列化（推送前端的前提）


if __name__ == "__main__":
    import sys
    sys.exit(U.run_module_tests(sys.modules[__name__]))