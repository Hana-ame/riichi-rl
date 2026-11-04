"""策略测试（mahjong/policies.py）——重点是 2026-11-04 对 GreedyTenpai 的修复回归。

运行：python -m tests.test_policies
"""

from __future__ import annotations

import unittest

import tests.utils as U
from mahjong import env as E
from mahjong import policies as POL
from mahjong import refengine as REF


def _shanten_of_discard(env, p_idx, tile):
    pl = env.players[p_idx]
    pl.hand.remove(tile)
    pl.hand.sort()
    try:
        sh = REF.ref_shanten(pl.full_counts())
    finally:
        pl.hand.append(tile)
        pl.hand.sort()
    return sh


class TestGreedyShantenMinimization(unittest.TestCase):
    """回归 T15：未听牌时 GreedyTenpai 必须按「向听数最小」选牌。

    修复前它只比「打完后待牌种数」，未听牌时该值恒 0，等于随机打牌
    （实测 13 张真听牌 40 局仅 5 次）。
    """

    def test_picks_min_shanten_discard(self):
        # 构造明确更优的牌型：两组对子 + 两面搭子，孤张字牌该先打
        hand = [0, 1, 2, 3, 4, 5, 9, 10, 18, 18, 19, 19, 27]  # 13 张
        env = U.make_env(seed=11, dealer=0)
        env.start_turn()  # P0 摸到第 14 张
        U.set_hand(env, 0, hand[:])
        # 把摸到的牌补回去，保持 14 张
        pl = env.players[0]
        if len(pl.hand) == 13:
            pl.hand.append(27)
            pl.hand.sort()
        legal = env.legal_actions(0)
        discards = [a for a in legal if a.type == E.ACT_DISCARD]
        self.assertGreater(len(discards), 3)
        pol = POL.make_policy("greedy_tenpai", seed=0)
        chosen = pol.choose(env, 0, legal)
        self.assertEqual(chosen.type, E.ACT_DISCARD)
        best = min(_shanten_of_discard(env, 0, a.tile) for a in discards)
        got = _shanten_of_discard(env, 0, chosen.tile)
        self.assertEqual(got, best,
                         f"应打向听最小的牌（best={best}, got={got}）")


class TestGreedyNoPrematureRyuukyoku(unittest.TestCase):
    """回归 T16：GreedyTenpai 不得主动宣言流局（听牌后应继续等自摸/荣和）。

    旧用例靠 4 局随机自对弈「碰巧遇到可宣言流局的机会」，但 GreedyTenpai
    一进听就立直，而引擎对立直者不生成 RYUUKYOKU 动作（立直后手牌锁定），
    机会数实测恒为 0，断言必然失败。改为构造确定性局面（非立直、摸切后
    听牌），同时把「14 张判听牌必须去掉 last_drawn」的引擎缺陷钉死在
    `tests/test_env_core.py::test_tenpai_check_uses_last_drawn`。
    """

    def _tenpai_env(self):
        env = U.make_env(seed=5, dealer=0)
        env.start_turn()
        # 123m 456m 789m 222p + 东 + 北（刚摸的北）。摸切北 -> 听东单骑。
        U.set_hand(env, 0,
                   [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 10, 10, 27, 28])
        env.players[0].last_drawn = 28
        return env

    def test_ryuukyoku_action_generated_when_tenpai(self):
        env = self._tenpai_env()
        kinds = {a.type for a in env.legal_actions(0)}
        self.assertIn(E.ACT_RYUUKYOKU, kinds,
                      "摸切后听牌必须生成流局宣言动作（修复前 0 次）")
        self.assertIn(E.ACT_NAGASHI, kinds)

    def test_policy_never_declares_ryuukyoku(self):
        env = self._tenpai_env()
        legal = env.legal_actions(0)
        self.assertTrue(any(a.type == E.ACT_RYUUKYOKU for a in legal),
                        "前置条件：该局面确实可宣言流局")
        pol = POL.make_policy("greedy_tenpai", seed=0)
        chosen = pol.choose(env, 0, legal)
        self.assertNotEqual(chosen.type, E.ACT_RYUUKYOKU,
                            "听牌后不得主动宣言流局（应继续摸打等和牌）")


class TestGreedyNoRandomCalls(unittest.TestCase):
    """回归 T17：响应阶段不得随机吃/碰（保持门前清才能立直）。"""

    def test_response_always_pass_unless_ron(self):
        seen_calls = 0
        made_calls = 0
        for seed in range(4):
            env = U.make_env(seed=seed, dealer=seed % 4)
            pols = {i: POL.make_policy("greedy_tenpai", seed=seed * 97 + i)
                    for i in range(4)}
            r = env.start_turn()
            steps = 0
            while r is None and steps < 600:
                steps += 1
                acts = {}
                for p in env.pending_players():
                    legal = env.legal_actions(p)
                    if not legal:
                        acts[p] = E.PASS_ACTION
                        continue
                    has_call = any(a.type in (E.ACT_CHI, E.ACT_PON, E.ACT_MINKAN)
                                   for a in legal)
                    a = pols[p].choose(env, p, legal)
                    if has_call:
                        seen_calls += 1
                        if a.type in (E.ACT_CHI, E.ACT_PON, E.ACT_MINKAN):
                            made_calls += 1
                    acts[p] = a
                r = env.step(acts)
        self.assertGreater(seen_calls, 0, "应出现过可吃/碰的机会")
        self.assertEqual(made_calls, 0, "响应阶段不得随机吃碰")


class TestGreedyRiichi(unittest.TestCase):
    """能立直时必须立直（立直=听牌锁定+必有役）。"""

    def test_riichi_preferred(self):
        riichi_times = 0
        for seed in range(5):
            env = U.make_env(seed=seed, dealer=seed % 4)
            pols = {i: POL.make_policy("greedy_tenpai", seed=seed * 97 + i)
                    for i in range(4)}
            r = env.start_turn()
            steps = 0
            while r is None and steps < 600:
                steps += 1
                acts = {}
                for p in env.pending_players():
                    legal = env.legal_actions(p)
                    if not legal:
                        acts[p] = E.PASS_ACTION
                        continue
                    has_ri = any(a.type == E.ACT_RIICHI for a in legal)
                    a = pols[p].choose(env, p, legal)
                    if has_ri:
                        riichi_times += 1
                        self.assertEqual(a.type, E.ACT_RIICHI,
                                         "有立直机会时应立直")
                    acts[p] = a
                r = env.step(acts)
        self.assertGreater(riichi_times, 0, "5 局内应出现过立直机会")


class TestBaselineBehaviors(unittest.TestCase):
    """其余基线的接口/行为冒烟。"""

    def test_random_returns_legal(self):
        env = U.make_env(seed=3)
        pol = POL.make_policy("random", seed=0)
        r = env.start_turn()
        steps = 0
        while r is None and steps < 100:
            steps += 1
            acts = {}
            for p in env.pending_players():
                legal = env.legal_actions(p)
                if not legal:
                    acts[p] = E.PASS_ACTION
                    continue
                a = pol.choose(env, p, legal)
                self.assertIn(a, legal)
                acts[p] = a
            r = env.step(acts)

    def test_all_policies_constructible(self):
        for name in ("random", "greedy_tenpai", "call_first", "yaochu_first"):
            pol = POL.make_policy(name, seed=0)
            self.assertIsNotNone(pol)


if __name__ == "__main__":
    import sys
    sys.exit(U.run_module_tests(sys.modules[__name__]))