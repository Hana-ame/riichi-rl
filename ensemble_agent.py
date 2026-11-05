"""多 ckpt logits 平均的集成 Agent（56 槽/47 槽通用，仅在合法槽上平均）。

动机（2026-11-06）：eff 教师线的模型已追平教师（同池 14.90% vs 15.52%），
剩余误差主要是 BC 复合误差（学生自走会漂）。集成不同 DAgger 轮次的权重
能压低单模型的策略方差，是低成本、不引入规则接管的提升手段。

用法：
    from ensemble_agent import EnsembleAgent
    ag = EnsembleAgent("ens", ["checkpoints/a.pt", "checkpoints/b.pt"])
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from version_benchmark import AutoAgent


class EnsembleAgent:
    def __init__(self, name, ckpts):
        self.name = name
        self.parts = [AutoAgent(f"{name}#{i}", c) for i, c in enumerate(ckpts)]
        base = self.parts[0]
        # 所有成员必须同架构/同槽位语义
        for p in self.parts[1:]:
            if (p.n_actions, p.mode, p.is47) != (base.n_actions, base.mode, base.is47):
                raise ValueError("EnsembleAgent 成员架构不一致")
        self.TH = base.TH
        self.enc = base.enc
        self.is47 = base.is47
        self.n_actions = base.n_actions
        self.mode = base.mode
        self.hstate = None

    def set_game(self, env, hstate=None):
        self.hstate = hstate
        for p in self.parts:
            p.set_game(env, hstate)

    def act(self, env, p, legal):
        ov = _rule_override(legal)
        if ov is not None:
            return ov
        base = self.parts[0]
        slot_fn = self.TH.slot_of if self.is47 else None
        from version_benchmark import slot_of54
        slot_fn = slot_fn or slot_of54
        if self.mode == "hanchan":
            vec, sh, sit, gu, tf = self.TH.encode(env, self.enc, self.hstate, p)
            legal2 = [a for a in legal if slot_fn(a) >= 0]
            slots = [slot_fn(a) for a in legal2]
            V = torch.from_numpy(vec).unsqueeze(0)
            T = torch.from_numpy(tf).unsqueeze(0)
            with torch.no_grad():
                lg = sum(part.model(V, T)[0].reshape(-1).numpy()
                         for part in self.parts) / len(self.parts)
        else:
            planes, scalar = self.enc.encode(env, p)
            vec = np.concatenate([planes.ravel(), scalar]).astype(np.float32)
            legal2 = list(legal)
            slots = [slot_fn(a) for a in legal2]
            V = torch.from_numpy(vec).unsqueeze(0)
            with torch.no_grad():
                lg = sum(part.model(V)[0].reshape(-1).numpy()
                         for part in self.parts) / len(self.parts)
        mask = np.full(self.n_actions, -1e9, dtype=np.float32)
        mask[slots] = 0.0
        pick = int((lg + mask).argmax())
        return legal2[slots.index(pick)] if pick in slots else legal2[0]


def _rule_override(legal):
    """与 AutoAgent 同口径：和牌/规则强制项优先。"""
    from version_benchmark import _rule_override as ro
    return ro(legal)
