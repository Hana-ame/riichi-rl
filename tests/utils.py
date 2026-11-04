"""测试公共工具。

所有测试都不依赖 pytest，`python -m tests.xxx` 或 `python tests/xxx.py` 可单独跑。
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mahjong import env as E  # noqa: E402


def make_env(seed=0, dealer=0):
    return E.MahjongEnv(seed=seed, dealer=dealer)


def set_hand(env, p_idx, tiles):
    """直接设定某玩家手牌（测试构造局面用），并清空副露/牌河。"""
    pl = env.players[p_idx]
    pl.hand = sorted(tiles)
    pl.melds = []
    pl.river = []
    pl.riichi = False
    pl.furiten = False


def hand_counts(tiles):
    c = [0] * 34
    for t in tiles:
        c[t] += 1
    return c


def playout(env, policies, max_steps=600, on_step=None):
    """跑完一局；policies: {seat: Policy}。返回 GameResult 或 None。"""
    r = env.start_turn()
    steps = 0
    while r is None and steps < max_steps:
        steps += 1
        acts = {}
        for p in env.pending_players():
            legal = env.legal_actions(p)
            if not legal:
                acts[p] = E.PASS_ACTION
                continue
            acts[p] = policies[p].choose(env, p, legal)
        r = env.step(acts)
        if on_step is not None:
            on_step(env, steps, r)
    return r


def run_module_tests(module):
    """统一入口：跑一个模块里的全部 unittest 用例。"""
    import unittest
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1