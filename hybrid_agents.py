"""混合代码-模型 Agent：打牌用代码算向听，其它决策交给神经网络。

用户建议（2026-11-05）：「混合代码的模型，比如打哪张牌向听多少用代码计算。」

思路：小模型（453K 参数）自己往往学不会稳定的牌效；但「打哪张能让向听最小」
是纯规则计算，可以直接用代码算。于是做**混合 Agent**：

- **打牌阶段**：用 `ref_shanten` 对每个候选弃牌算「打完后向听数」，选最小；
  平手时用模型对该弃牌的 logit 做 tie-break（模型仍参与牌风判断）；
- **其它阶段**（立直/默听/和牌/吃碰杠/响应）：交给模型 argmax（带合法 mask）；
- 和牌/荣和等强制合法动作沿用规则优先级（与 arena_elo 的 Agent 一致）。

支持 47 槽（半庄线）与 54 槽（obs2 线）ckpt，架构/特征模式从 ckpt 自动推断。

用法：
    from hybrid_agents import HybridShantenAgent
    agent = HybridShantenAgent("hybrid:teacher47_v3", "checkpoints/teacher47_v3.pt")
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from arena_elo import Agent, _rule_override
from mahjong import env as E
from mahjong import obs2 as O2
from mahjong import refengine as REF
from mahjong.net import MahjongNet
from train_reinforce import slot_of as slot_of54, N_ACTIONS as NA54


class HybridShantenAgent(Agent):
    """代码算向听选弃牌 + 模型决定其它动作。"""

    kind = "hybrid"

    def __init__(self, name, ckpt):
        import train_hanchan as TH
        self.TH = TH
        self.name = name
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck)
        n_actions = int(sd["policy.weight"].shape[0])
        ch = int(ck.get("channels", 160))
        nb = int(ck.get("blocks", 6))
        head_in = int(sd["head_fc.0.weight"].shape[1])
        in_scalar = head_in - ch * 5
        extra_total = in_scalar - 26
        if int(ck.get("touch_dim", 0)) or extra_total > 0:
            touch = int(ck.get("touch_dim", 8))
            extra = int(ck.get("extra_scalar", ck.get("sit_dim", 12)))
            if touch + extra != extra_total:
                touch, extra = 8, extra_total - 8
            self.mode = "hanchan"
        else:
            touch, extra = 0, 0
            self.mode = "obs2"
        self.model = MahjongNet(channels=ch, n_blocks=nb, n_actions=n_actions,
                                touch_dim=touch, extra_scalar=extra)
        self.model.load_state_dict(sd)
        self.model.eval()
        self.enc = O2.ObsEncoder()
        self.hstate = None
        self.n_actions = n_actions
        self.is47 = n_actions == 47

    def set_game(self, env, hstate=None):
        self.hstate = hstate

    # ---- 模型前向 ----
    def _logits(self, env, p, legal):
        slot_fn = self.TH.slot_of if self.is47 else slot_of54
        if self.mode == "hanchan":
            vec, sh, sit, gu, tf = self.TH.encode(env, self.enc, self.hstate, p)
            with torch.no_grad():
                logits = self.model(torch.from_numpy(vec).unsqueeze(0),
                                    torch.from_numpy(tf).unsqueeze(0))[0]
        else:
            planes, scalar = self.enc.encode(env, p)
            vec = np.concatenate([planes.ravel(), scalar]).astype(np.float32)
            with torch.no_grad():
                logits = self.model(
                    torch.from_numpy(vec).unsqueeze(0))[0]
        return logits.reshape(-1).numpy(), [slot_fn(a) for a in legal]

    # ---- 代码牌效 ----
    def _shanten_after(self, env, p_idx, tile):
        p = env.players[p_idx]
        p.hand.remove(tile)
        p.hand.sort()
        try:
            return int(REF.ref_shanten(p.full_counts()))
        finally:
            p.hand.append(tile)
            p.hand.sort()

    def act(self, env, p, legal):
        if not legal:
            return E.PASS_ACTION
        # 和牌/荣和等强制动作先走规则，避免模型漏和
        ov = _rule_override(legal)
        if ov is not None:
            return ov

        logits, slots = self._logits(env, p, legal)

        # 打牌阶段：代码算向听最小，模型 logit 做 tie-break
        if env.phase == E.Phase.DISCARD:
            discs = [(i, a) for i, a in enumerate(legal)
                     if a.type == E.ACT_DISCARD]
            if discs:
                best = None
                best_key = None
                for i, a in discs:
                    if a.tile is None or a.tile not in env.players[p].hand:
                        continue
                    sh = self._shanten_after(env, p, a.tile)
                    slot = self.TH.slot_of(a) if self.is47 else slot_of54(a)
                    model_score = float(logits[slot]) if 0 <= slot < len(logits) \
                        else -1e9
                    key = (sh, -model_score)
                    if best_key is None or key < best_key:
                        best_key = key
                        best = i
                if best is not None:
                    return legal[best]

        # 其它动作：模型 masked argmax（slot >= 0 的合法动作里选）
        mask = np.full(self.n_actions, -1e9, dtype=np.float32)
        ok = [(i, s) for i, s in zip(range(len(legal)), slots) if s >= 0]
        if not ok:
            return legal[0]
        for i, s in ok:
            mask[s] = 0.0
        pick_idx, pick_slot = max(ok, key=lambda x: logits[x[1]] + mask[x[1]])
        return legal[pick_idx]