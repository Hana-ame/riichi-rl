"""dora 策略（硬门槛项）：推理期混合打分 —— 模型 logits + 保留宝牌/赤牌的加权。

用户明确：dora 对应策略是硬性达标项；本实现不改模型权重，而是「模型打分 + 代码规则」
混合（与之前混合 Agent 同思路）：
  score(action) = logit(action) - alpha * dora_discard_penalty(tile)
其中 dora_discard_penalty=1 若该弃牌是宝牌/赤牌（打掉即损失打点），0 否则。
alpha=0 即基线。
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from mahjong import env as E
from version_benchmark import AutoAgent
from train_hanchan import encode, slot_of, N_ACTIONS, SIT_DIM
from mahjong import obs2 as O2
from mahjong.tiles import NUM_TILES


class DoraAgent(AutoAgent):
    def __init__(self, name, ckpt, alpha=1.0):
        super().__init__(name, ckpt)
        self.alpha = float(alpha)
        self.enc = O2.ObsEncoder()

    def _dora_tiles(self, env):
        d = set()
        if env.dora_indicator is not None:
            d.add((env.dora_indicator + 1) % NUM_TILES)
        return d

    def act(self, env, p, legal):
        a = super().act(env, p, legal)
        if self.alpha <= 0 or not env.legal_actions(p):
            return a
        discards = [x for x in legal if getattr(x, "kind", "") == "discard"]
        if not discards:
            return a
        dora = self._dora_tiles(env)
        if not any((x.tile in dora) for x in discards):
            return a
        vec, sh, sit, gu, tf = encode(env, self.enc, self.hstate, p)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(vec)[None], torch.from_numpy(tf)[None])[0][0].numpy()
        slots = [slot_of(x) for x in discards]
        best, best_s = None, -1e18
        for x, s in zip(discards, slots):
            sc = float(logits[s]) - self.alpha * (1.0 if x.tile in dora else 0.0)
            if sc > best_s:
                best_s, best = sc, x
        # 只在模型原本会打宝牌时覆盖；否则保持模型动作（避免全局扰动）
        if getattr(a, "tile", None) in dora:
            return best
        return a
