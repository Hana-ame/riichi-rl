"""危险张接入主策略（模型版）：弃牌 = 模型 logits − λ·危险分。

危险分来自 `opponent_model.py`（训练出来的对手持牌概率，不是手写筋/现物表）：
danger[t] = Σ_对手 威胁权重(立直1.0/未立直0.25) × P(对手持有 t)。
对所有合法弃牌统一打分，不再用「向听优先」规则覆盖模型（区别于 hybridd）。

用法：
  python eval_selfplay_gate.py --danger-ckpt teacher47_eff_v3 --danger-lam 5 --deals 60
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from version_benchmark import AutoAgent, _rule_override
from mahjong import env as E
from mahjong.tiles import NUM_TILES

ROOT = os.path.dirname(os.path.abspath(__file__))


class ModelDangerAgent(AutoAgent):
    def __init__(self, name, ckpt, lam=5.0, opp_ckpt=None, riichi_only=False):
        super().__init__(name, ckpt)
        self.lam = float(lam)
        self.riichi_only = bool(riichi_only)
        self.opp = None
        self.opp_ckpt = opp_ckpt or os.path.join(ROOT, "checkpoints", "opponent_model.pt")
        if os.path.exists(self.opp_ckpt):
            from opponent_model import OpponentNet
            ck = torch.load(self.opp_ckpt, map_location="cpu", weights_only=False)
            self.opp = OpponentNet()
            self.opp.load_state_dict(ck["state_dict"])
            self.opp.eval()

    def _opp_hold(self, env, p_idx):
        from opponent_model import build_features
        out = {}
        me = env.players[p_idx]
        hand = np.asarray(me.counts(), dtype=np.int8)
        left = len(env.wall) - env.wall_idx
        di = env.dora_indicator
        dora = int(di) if (di is not None and di >= 0) else -1
        for t in range(4):
            if t == p_idx:
                continue
            tp = env.players[t]
            target_river = np.zeros(NUM_TILES, dtype=np.int8)
            other_rivers = np.zeros(NUM_TILES, dtype=np.int8)
            for tile in tp.river:
                target_river[tile] += 1
            for j in range(4):
                if j in (p_idx, t):
                    continue
                for tile in env.players[j].river:
                    other_rivers[tile] += 1
            f = build_features(hand, target_river, other_rivers, len(tp.melds),
                               1 if tp.riichi else 0, int(target_river.sum()),
                               left, dora)
            with torch.no_grad():
                out[t] = self.opp(torch.from_numpy(f).unsqueeze(0))[0].numpy()
        return out

    def danger(self, env, p_idx):
        d = np.zeros(NUM_TILES, dtype=np.float32)
        for t, pr in self._opp_hold(env, p_idx).items():
            w = 1.0 if env.players[t].riichi else 0.25
            d += w * np.asarray(pr)
        return d

    def act(self, env, p, legal):
        ov = _rule_override(legal)
        if ov is not None:
            return ov
        if self.lam <= 0 or self.opp is None:
            return super().act(env, p, legal)
        from train_hanchan import slot_of
        from version_benchmark import slot_of54
        slot_fn = self.TH.slot_of if self.is47 else slot_of54
        vec, sh, sit, gu, tf = self.TH.encode(env, self.enc, self.hstate, p)
        legal2 = [a for a in legal if slot_fn(a) >= 0]
        slots = [slot_fn(a) for a in legal2]
        with torch.no_grad():
            logits = self.model(torch.from_numpy(vec).unsqueeze(0),
                                torch.from_numpy(tf).unsqueeze(0))[0]
        lg = logits.reshape(-1).numpy().astype(np.float32).copy()
        if env.phase == E.Phase.DISCARD:
            if self.riichi_only and not any(
                    q.riichi for q in env.players if q.idx != p):
                return super().act(env, p, legal)
            d = self.danger(env, p)
            m = float(d.max()) if d.size else 0.0
            if m > 0:
                d = d / m
            for a in legal2:
                if a.type == E.ACT_DISCARD:
                    s = slot_fn(a)
                    lg[s] -= self.lam * float(d[a.tile])
        mask = np.full(self.n_actions, -1e9, dtype=np.float32)
        mask[slots] = 0.0
        pick = int((lg + mask).argmax())
        return legal2[slots.index(pick)] if pick in slots else legal2[0]


def make_danger_agent(name, ckpt, lam):
    return ModelDangerAgent(name, ckpt, lam=lam)
