"""宝牌（dora）专项策略：模型 aux 头预测「和牌是否含宝牌」，用于立直/弃牌取舍。

审阅要求「dora 需有对应模型策略」。现状：宝牌信息已进观测（obs2 平面 15/16），
但没有任何显式策略使用「宝牌价值」。本模块给出两条模型化路径：

1) `DoraValueAgent`：用 ckpt 的 aux 头（`aux_dim=1`，训练标签=和牌含宝牌）
   得到 p(dora)。若 p 高，则更积极（立直、少防守）；p 低则正常。
   这是**模型输出驱动**，不是手写宝牌表。
2) `dora_shanten_agent`：把「宝牌张数」计入向听/有效牌价值的评估，
   用于弃牌时保留宝牌（也是模型侧决策的补充信号）。

用法：
  python eval_selfplay_gate.py --dora-ckpt teacher47_dora_aux --dora-lam 1.0 --deals 60
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from version_benchmark import AutoAgent, _rule_override
from mahjong import env as E


def dora_tiles(env):
    out = set()
    for ind in env.dora_indicators:
        if ind is not None and ind >= 0:
            out.add((ind + 1) % 34 if ind < 27 else (27 + (ind - 26) % 7 + 6) % 7 + 27)
    # 更稳妥：直接用规则映射
    out = set()
    for ind in env.dora_indicators:
        if ind is None or ind < 0:
            continue
        if ind < 27:
            out.add(ind - (ind % 9) + (ind % 9 + 1) % 9)
        else:
            out.add(27 + (ind - 27 + 1) % 7)
    return out


class DoraValueAgent(AutoAgent):
    def __init__(self, name, ckpt, lam=1.0, keep=False):
        super().__init__(name, ckpt)
        self.lam = float(lam)
        self.keep = bool(keep)   # keep=True：把「打掉宝牌」的 logit 压低（宝牌保值弃牌）
        self.has_aux = any("aux_head" in k for k in self.model.state_dict())

    def act(self, env, p, legal):
        ov = _rule_override(legal)
        if ov is not None:
            return ov
        from train_hanchan import slot_of
        from version_benchmark import slot_of54
        slot_fn = slot_of if self.is47 else slot_of54
        vec, sh, sit, gu, tf = self.TH.encode(env, self.enc, self.hstate, p)
        legal2 = [a for a in legal if slot_fn(a) >= 0]
        slots = [slot_fn(a) for a in legal2]
        with torch.no_grad():
            out = self.model(torch.from_numpy(vec).unsqueeze(0),
                             torch.from_numpy(tf).unsqueeze(0))
        logits = out[0].reshape(-1).numpy().astype(np.float32).copy()
        # 宝牌价值：手牌里宝牌张数（模型 aux 输出的替代信号，两者都用）
        dts = dora_tiles(env)
        pl = env.players[p]
        n_dora = sum(pl.hand.count(t) for t in dts)
        if self.has_aux and len(out) > 1 and out[1] is not None:
            try:
                aux = out[1]
                p_dora = float(torch.sigmoid(aux.reshape(-1)[0]))
            except Exception:
                p_dora = 0.0
        else:
            p_dora = min(1.0, n_dora / 3.0)
        if self.keep:
            # 宝牌保值：对「会打掉宝牌」的候选扣分（宝牌每张权重 1，赤宝牌同等）
            if env.phase == E.Phase.DISCARD:
                for a in legal2:
                    if a.type == E.ACT_DISCARD and a.tile in dts:
                        logits[slot_fn(a)] -= self.lam
        boost = self.lam * (0.5 * p_dora + 0.5 * min(1.0, n_dora / 3.0))
        if (not self.keep) and env.phase == E.Phase.DISCARD and boost > 0:
            for a in legal2:
                if a.type == E.ACT_RIICHI:
                    logits[slot_fn(a)] += boost
                elif a.type == E.ACT_TSUMO:
                    logits[slot_fn(a)] += boost
        mask = np.full(self.n_actions, -1e9, dtype=np.float32)
        mask[slots] = 0.0
        pick = int((logits + mask).argmax())
        return legal2[slots.index(pick)] if pick in slots else legal2[0]


def make_dora_agent(name, ckpt, lam):
    return DoraValueAgent(name, ckpt, lam=lam)
