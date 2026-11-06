"""让模型自己决定「立直 vs 继续打」——去掉 `_rule_override` 里的立直强制。

发现（2026-11-06）：`arena_elo._rule_override()` 对**所有**合法动作先返回
TSUMO/RON，再返回 **RIICHI**。也就是说，此前所有 AutoAgent 评测里
「能不能立直就立直」是**规则接管**，模型从没做过立直决策——
这既是审阅说的「规则接管」之一，也直接压低了模型的可学习空间
（默听/立直是收益差异很大的决策）。

本 Agent：和牌（TSUMO/RON）仍规则必和（所有 AI 同口径），
**立直改成由模型 logits 与弃牌 logits 比较后决定**。

用法：
  python eval_selfplay_gate.py --model-choice-ckpt teacher47_eff_wuke --deals 240
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from version_benchmark import AutoAgent
from mahjong import env as E


class ModelChoiceAgent(AutoAgent):
    def act(self, env, p, legal):
        # 只保留「和牌必和」，不再强制立直
        for a in legal:
            if a.type in (E.ACT_TSUMO, E.ACT_RON):
                return a
        from train_hanchan import slot_of
        from version_benchmark import slot_of54
        slot_fn = slot_of if self.is47 else slot_of54
        vec, sh, sit, gu, tf = self.TH.encode(env, self.enc, self.hstate, p)
        legal2 = [a for a in legal if slot_fn(a) >= 0]
        if not legal2:
            return legal[0]
        slots = [slot_fn(a) for a in legal2]
        with torch.no_grad():
            logits = self.model(torch.from_numpy(vec).unsqueeze(0),
                                torch.from_numpy(tf).unsqueeze(0))[0]
        lg = logits.reshape(-1).numpy()
        mask = np.full(self.n_actions, -1e9, dtype=np.float32)
        mask[slots] = 0.0
        pick = int((lg + mask).argmax())
        return legal2[slots.index(pick)] if pick in slots else legal2[0]
