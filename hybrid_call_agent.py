"""Hybrid 副露 Agent：叫牌用 Tenhou BC（会叫 chi/pon），弃牌用 eff_wuke_d2。

背景：eff 系列从不叫牌（教师 GreedyTenpai 从不叫），Tenhou BC 会叫
（训练数据里 4718 chi / 509 pon）。这个 Agent 组合两者：
- 弃牌：eff_wuke_d2（当前 best）
- 叫牌：Tenhou BC 的 logits 决定
- 和牌：规则必和
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mahjong import env as E
from train_hanchan import slot_of, N_ACTIONS
from version_benchmark import make_agent


class HybridCallAgent:
    """叫牌=Tenhou BC，弃牌=eff_wuke_d2。"""

    def __init__(self, name, call_ckpt, disc_ckpt):
        self.name = name
        self.call_ag = make_agent("call", call_ckpt)
        self.disc_ag = make_agent("disc", disc_ckpt)
        self.hstate = None
        self.calls = {"chi": 0, "pon": 0, "kan": 0}

    def set_game(self, env, hstate=None):
        self.hstate = hstate
        self.call_ag.set_game(env, hstate)
        self.disc_ag.set_game(env, hstate)

    def act(self, env, p, legal):
        # 和牌必和
        for wt in (E.ACT_TSUMO, E.ACT_RON):
            ws = [a for a in legal if a.type == wt]
            if ws:
                return ws[0]
        # 响应阶段：Tenhou BC 决定叫牌
        if env.phase == E.Phase.RESPONSE:
            calls = [a for a in legal
                     if a.type in (E.ACT_CHI, E.ACT_PON,
                                   E.ACT_MINKAN, E.ACT_ANKAN, E.ACT_ADD_KAN)]
            if calls:
                a = self.call_ag.act(env, p, legal)
                if a.type == E.ACT_CHI:
                    self.calls["chi"] += 1
                    return a
                elif a.type == E.ACT_PON:
                    self.calls["pon"] += 1
                    return a
                elif a.type in (E.ACT_MINKAN, E.ACT_ANKAN, E.ACT_ADD_KAN):
                    self.calls["kan"] += 1
                    return a
            return E.PASS_ACTION if E.PASS_ACTION in legal else legal[0]
        # 弃牌阶段：eff_wuke_d2
        return self.disc_ag.act(env, p, legal)
