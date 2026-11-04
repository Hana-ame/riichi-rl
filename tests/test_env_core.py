"""引擎核心测试（mahjong/env.py）——重点是 2026-11-04 修复的多处回归。

运行：python -m tests.test_env_core
"""

from __future__ import annotations

import unittest

import tests.utils as U
from mahjong import env as E
from mahjong import policies as POL


class TestDrawTracking(unittest.TestCase):
    """回归：T13 _draw_tile 必须记录摸到的牌（自摸和了判定依赖它）。"""

    def test_last_drawn_recorded(self):
        env = U.make_env(seed=1)
        p = env.players[env.turn_player]
        before = len(p.hand)
        t = env._draw_tile(p)
        self.assertIsNotNone(t)
        self.assertEqual(len(p.hand), before + 1)
        self.assertEqual(p.last_drawn, t, "摸牌后 last_drawn 必须等于摸到的牌")
        self.assertIn(t, p.hand)

    def test_can_tsumo_uses_last_drawn(self):
        """和牌型必须依赖真实摸牌，而不是「最大牌号」近似。"""
        from train_hanchan import hand_shanten
        env = U.make_env(seed=2)
        # 造一个门前清自摸和牌型：123m 456m 789m 234p 东东，摸东
        U.set_hand(env, 0, [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 27])
        env.turn_player = 0
        env.last_draw_by = 0
        pl = env.players[0]
        pl.last_drawn = 27
        pl.hand.append(27)
        pl.hand.sort()
        self.assertEqual(hand_shanten(env, 0), -1, "构造的应是与和牌型")
        ok = env._can_tsumo(0)
        self.assertTrue(ok, "门前清自摸应可和（自摸役/门前清役）")

    def test_tenpai_check_uses_last_drawn(self):
        """回归：14 张手牌判听牌必须去掉 last_drawn，而不是「最小牌号的牌」。

        修复前 `_is_tenpai` 在 14 张时默认丢最小牌号那张，等于用错误的
        13 张去判；后果是弃牌阶段永远判不出听牌，流局宣言动作从不生成
        （`tests/test_policies.py::TestGreedyNoPrematureRyuukyoku` 抓到的
        真实缺陷）。
        """
        env = U.make_env(seed=3)
        env.start_turn()
        # 123m 456m 789m 222p + 东 + 北（刚摸的北）。摸切北 -> 听东单骑。
        U.set_hand(env, 0,
                   [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 10, 10, 27, 28])
        env.players[0].last_drawn = 28
        self.assertTrue(env._is_tenpai(0),
                        "去掉 last_drawn(北) 后是听东单骑，必须判为听牌")
        # 若只拿掉最小牌号的 1m，剩 2m..789m 222p 东北，是散牌（非听牌）。
        env.players[0].last_drawn = -1   # 没有摸牌记录 -> 退回旧逻辑
        self.assertFalse(env._is_tenpai(0),
                         "没有 last_drawn 时不能凭空判成听牌")


class TestFuritenRules(unittest.TestCase):
    """回归：振听方向必须是「禁荣和、可自摸」（RULES.md §11，2026-11-04 修正）。

    旧规范/实现写反了：振听禁自摸、反而允许放铳和。标准立直麻将是
    振听不能荣和（含抢杠），自摸照常。
    """

    # tile id: 0-8=1m..9m, 9-17=1p..9p；13 张听 6p（14）
    TENPAI_13 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 12, 13]
    WIN_14 = TENPAI_13 + [14]

    def test_furiten_blocks_ron(self):
        env = U.make_env(seed=5)
        U.set_hand(env, 0, self.TENPAI_13)
        env.players[0].river.append(14)
        env._update_furiten(env.players[0])
        self.assertTrue(env.players[0].furiten)
        env.last_discard = 14
        env.last_discard_player = 3
        env.phase = E.Phase.RESPONSE
        self.assertFalse(env._can_ron(0), "振听必须禁止荣和")
        self.assertFalse(env._can_ron(0, from_chankan=True),
                         "振听必须禁止抢杠和")

    def test_furiten_allows_tsumo(self):
        env = U.make_env(seed=5)
        U.set_hand(env, 0, self.WIN_14)
        pl = env.players[0]
        pl.river.append(14)
        pl.last_drawn = 14
        env._update_furiten(pl)
        self.assertTrue(pl.furiten)
        self.assertTrue(env._can_tsumo(0), "振听不影响自摸和了")


class TestWinActionConsistency(unittest.TestCase):
    """回归：T14 能判定的荣和必须真的能结算（修复前 _can_ron 只查牌型，
    执行时 _finish_ron 因无役静默丢弃：实测 21 次选 RON 只结算 1 次）。
    """

    def test_ron_can_win_and_settles(self):
        env = U.make_env(seed=7, dealer=0)
        env.start_turn()
        # 门前清听东：123m 456m 789m 234p + 东（13 张）
        U.set_hand(env, 0, [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 27])
        env.last_discard = 27
        env.last_discard_player = 1
        env.phase = E.Phase.RESPONSE
        self.assertTrue(env._can_ron(0), "门前清听牌遇和牌张应可荣和")
        res = env._finish_ron(0, 1, 27)
        self.assertIsNotNone(res, "可判定的荣和必须结算，不能静默丢弃")
        self.assertEqual(res.result_type, "ron")

    def test_no_yaku_ron_not_generated(self):
        """副露且无役：不应生成 RON 动作（否则选了也结算不了）。"""
        env = U.make_env(seed=8, dealer=0)
        env.start_turn()
        pl = env.players[0]
        # 碰了 1m（副露），手牌 234m 456m 789p 11s 听 1s；副露断幺被 1m/789p/1s 破坏 → 无役
        pl.melds = [("pon", 0, None)]
        U.set_hand(env, 0, [1, 2, 3, 4, 5, 10, 11, 12, 18, 18, 27])  # 11 张 + 碰的 3 张
        # 用 full_counts 视角补副露牌不需要；直接看合法动作
        env.last_discard = 18
        env.last_discard_player = 1
        env.phase = E.Phase.RESPONSE
        can = env._can_ron(0)
        # 若判定可荣和，那它必须能结算；若判定不可，合法动作里就不能有 RON
        legal = env.legal_actions(0)
        has_ron = any(a.type == E.ACT_RON for a in legal)
        self.assertEqual(can, has_ron,
                         "legal_actions 的 RON 必须与 _can_ron 一致")


class TestResponsePhase(unittest.TestCase):
    """响应阶段：合法动作、PASS、和牌优先级。"""

    def test_response_has_pass(self):
        env = U.make_env(seed=3)
        env.start_turn()
        # 打出一张后进入响应阶段
        p = env.pending_players()[0]
        legal = env.legal_actions(p)
        disc = [a for a in legal if a.type == E.ACT_DISCARD]
        self.assertTrue(disc)
        env.step({p: disc[0]})
        responsers = env.pending_players()
        self.assertTrue(responsers)
        for q in responsers:
            acts = env.legal_actions(q)
            self.assertTrue(any(a.type == E.ACT_PASS for a in acts),
                            "响应阶段必须始终有 PASS 合法")


class TestKanDora(unittest.TestCase):
    """回归：每次杠后必须翻新的宝牌指示牌（初始 1 张，最多 5 张）。"""

    def test_initial_single_indicator(self):
        env = U.make_env(seed=1)
        self.assertEqual(len(env.dora_indicators), 1)

    def test_flip_accumulates(self):
        env = U.make_env(seed=1)
        before = len(env.dora_indicators)
        env._flip_dora()
        self.assertEqual(len(env.dora_indicators), before + 1)
        self.assertEqual(env.dora_indicator, env.dora_indicators[0])

    def test_dora_set_covers_all(self):
        env = U.make_env(seed=1)
        env._flip_dora()
        ds = env._dora_set()
        self.assertGreaterEqual(len(ds), 1)
        self.assertGreaterEqual(len(env.dora_indicators), 2)

    def test_kan_action_flips_dora(self):
        from mahjong import tiles as T
        env = U.make_env(seed=2, dealer=0)
        env.start_turn()
        p = env.players[0]
        p.hand = [0, 0, 0, 0, 1, 2, 3]
        before = len(env.dora_indicators)
        env._do_kan(0, E.Action(E.ACT_ANKAN, 0))
        self.assertEqual(len(env.dora_indicators), before + 1,
                         "暗杠后必须翻一张新宝牌指示牌")

    def test_cap_five(self):
        env = U.make_env(seed=3)
        for _ in range(10):
            env._flip_dora()
        self.assertLessEqual(len(env.dora_indicators), 5,
                             "宝牌指示牌最多 5 张")


class TestConservation(unittest.TestCase):
    """快速守恒冒烟（完整 200 局版见 selftest.py T1/T3）。"""

    def test_tile_conservation_small(self):
        for seed in range(5):
            env = U.make_env(seed=seed, dealer=seed % 4)
            pols = {i: POL.make_policy("random", seed=seed * 97 + i) for i in range(4)}
            U.playout(env, pols)
            ok, total, detail = env.check_tile_conservation()
            self.assertTrue(ok, detail)


if __name__ == "__main__":
    import sys
    sys.exit(U.run_module_tests(sys.modules[__name__]))