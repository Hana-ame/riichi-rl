"""Stage 2-lite：把预测辅助头并入策略决策（推理时，不改网络输入）。

做法：主策略 47 槽 argmax 弃牌；在弃牌阶段用辅助头算
  danger[tile] = Σ_opp  P_tenpai(opp) × P(opp 持有 tile)
其中 P_tenpai 来自 aux_heads（三头模型，取 opp_tenpai 头），
P(持有) 来自 opponent_model；对 logits 做 danger 惩罚后再 argmax。
立直/和了/副露仍用原策略（规则兜底）。

用法：make_agent("auxd:teacher47_100h_v2") 或在版本评测里用 "auxd:" 前缀。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
from arena_elo import CKPT

N_TILES = 34


class AuxDangerAgent:
    kind = "aux_danger"

    def __init__(self, name, ckpt, aux_ckpt=None, opp_ckpt=None, lam=0.8):
        from version_benchmark import make_agent
        from aux_heads import AuxHeads
        from opponent_model import OpponentNet, build_features
        self.build_features = build_features
        base_name = name.split(":", 1)[1] if ":" in name else name
        self.main = make_agent(base_name, ckpt)
        self.hstate = None
        self.lam = lam
        self.aux = None
        ap = aux_ckpt or os.path.join(CKPT, "aux_heads.pt")
        if os.path.exists(ap):
            ck = torch.load(ap, map_location="cpu", weights_only=False)
            self.aux = AuxHeads(ck["in_dim"])
            self.aux.load_state_dict(ck["state_dict"]); self.aux.eval()
        self.opp = None
        op = opp_ckpt or os.path.join(CKPT, "opponent_model.pt")
        if os.path.exists(op):
            ck = torch.load(op, map_location="cpu", weights_only=False)
            self.opp = OpponentNet()
            sd = ck.get("state_dict", ck)
            self.opp.load_state_dict(sd)
            self.opp.eval()

    def set_game(self, env, hstate):
        self.hstate = hstate
        if hasattr(self.main, "set_game"):
            self.main.set_game(env, hstate)

    def _danger(self, env, p_idx):
        if self.aux is None or self.opp is None:
            return np.zeros(N_TILES, dtype=np.float32)
        me = env.players[p_idx]
        hand = np.asarray(me.counts(), dtype=np.int8)
        danger = np.zeros(N_TILES, dtype=np.float32)
        for t in range(4):
            if t == p_idx:
                continue
            tp = env.players[t]
            tgt = np.zeros(N_TILES, dtype=np.int8)
            oth = np.zeros(N_TILES, dtype=np.int8)
            for tile in tp.river:
                tgt[tile] += 1
            for j in range(4):
                if j in (p_idx, t):
                    continue
                for tile in env.players[j].river:
                    oth[tile] += 1
            f = self.build_features(hand, tgt, oth, len(tp.melds),
                                    1 if tp.riichi else 0, int(tgt.sum()),
                                    len(env.wall) - env.wall_idx,
                                    int(env.dora_indicator) if env.dora_indicator is not None else -1)
            with torch.no_grad():
                _, p_tp, _ = self.aux(torch.from_numpy(f)[None])
                p_hold = self.opp(torch.from_numpy(f)[None])[0].numpy()
            pt = float(torch.sigmoid(p_tp)[0])
            danger += pt * np.asarray(p_hold, dtype=np.float32)
        return danger

    def act(self, env, p_idx, legal):
        from mahjong import env as E
        discards = [a for a in legal if a.type == E.ACT_DISCARD]
        if not discards or self.aux is None:
            return self.main.act(env, p_idx, legal)
        # 复用主策略的 logits（AutoAgent 有 model/enc/TH）
        m = getattr(self.main, "model", None)
        if m is None:
            return self.main.act(env, p_idx, legal)
        import train_hanchan as TH
        vec, sh, sit, gu, tf = TH.encode(env, self.main.enc, self.hstate, p_idx)
        with torch.no_grad():
            out = self.main.model(torch.from_numpy(vec)[None],
                                  torch.from_numpy(tf)[None])
            logits = (out[0] if isinstance(out, tuple) else out)[0].numpy()
        danger = self._danger(env, p_idx)
        slots = [TH.slot_of(a) for a in discards]
        scores = np.array([logits[s] - self.lam * float(danger[a.tile])
                           for a, s in zip(discards, slots)])
        return discards[int(np.argmax(scores))]
