"""P3-v2：用 Suphx 的 chi/pon/kan 三个头当我们模型的「叫牌教师」。

Suphx 的叫牌输入 = 22 维候选特征 + 291 维状态特征 = 313：
- 候选特征：is_dora(1) + is_aka(1) + 手牌(4) + 副露(16)
- 决策：score = sigmoid(model(x)) / threshold / 2，取最大；score<0.5 即 pass
  （等价于 sigmoid ≥ threshold，阈值来自 ckpt）

我们引擎不区分赤牌、也没有天凤 tile-id，这里用 tile type 近似（赤=5 的 type）。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(ROOT, "external", "Mahjong-AI")
NUM_TILES = 34
UPPER = np.tril(np.ones(4))


def _dora_types(hstate):
    # 由 hstate 的宝牌指示牌在 env 里拿不到，退化为按 env.dora_indicators
    return set()


def _hand_feature(counts):
    f = np.zeros((4, NUM_TILES), dtype=np.float32)
    for t, c in enumerate(counts):
        if c > 0:
            f[:, t] = UPPER[c - 1]
    return f


def _furo_feature(melds):
    feats = []
    for kind, tile, _arg in melds:
        f = np.zeros((4, NUM_TILES), dtype=np.float32)
        if kind == "chi":
            f[0, [tile, tile + 1, tile + 2]] = 1.0
        elif kind == "pon":
            f[[0, 1, 2], tile] = 1.0
        else:
            f[:, tile] = 1.0
        feats.append(f)
    while len(feats) < 4:
        feats.append(np.zeros((4, NUM_TILES), dtype=np.float32))
    return np.concatenate(feats[:4], axis=0)


def candidate_feature(env, p_idx, act, state_feat):
    p = env.players[p_idx]
    from mahjong import env as E
    counts = np.asarray(p.counts(), dtype=np.int32)
    dora = set(getattr(env, "_dora_types_cache", [])) if hasattr(env, "_dora_types_cache") else set()
    called = -1
    melds = list(p.melds)
    if act.type == E.ACT_CHI:
        called = act.arg
        start = act.tile
        need = [t for t in (start, start + 1, start + 2) if t != called]
        for t in need:
            counts[t] -= 1
        melds = melds + [("chi", start, -1)]
    elif act.type == E.ACT_PON:
        called = act.tile
        counts[called] -= 2
        melds = melds + [("pon", called, -1)]
    elif act.type == E.ACT_MINKAN:
        called = act.tile
        counts[called] -= 3
        melds = melds + [("minkan", called, -1)]
    elif act.type == E.ACT_ANKAN:
        called = act.tile
        counts[called] = 0
        melds = melds + [("ankan", called, -1)]
    is_dora = np.ones((1, NUM_TILES), dtype=np.float32) * (1.0 if called in dora else 0.0)
    is_aka = np.ones((1, NUM_TILES), dtype=np.float32) * (1.0 if called in (4, 13, 22) else 0.0)
    cand = np.concatenate([is_dora, is_aka, _hand_feature(counts),
                           _furo_feature(melds)], axis=0)
    return np.concatenate([cand, state_feat], axis=0)   # 22 + 291


class SuphxCallTeacher:
    def __init__(self):
        sys.path.insert(0, EXT)
        from model.models import FuroModel
        self.heads = {}
        for kind in ("chi", "pon", "kan"):
            ck = torch.load(os.path.join(EXT, "model", "saved",
                                         f"{kind}-model", "best.pt"),
                            map_location="cpu", weights_only=False)
            m = FuroModel(in_channels=ck["in_channels"],
                          num_layers=ck["num_layers"])
            m.load_state_dict(ck["state_dict"]); m.eval()
            self.heads[kind] = (m, float(ck["threshold"]))
        sys.path.remove(EXT)

    def decide(self, env, p_idx, legal, state_feat):
        from mahjong import env as E
        best = None
        for a in legal:
            if a.type in (E.ACT_PON, E.ACT_CHI):
                kind = "pon" if a.type == E.ACT_PON else "chi"
            elif a.type in (E.ACT_MINKAN, E.ACT_ANKAN, E.ACT_ADD_KAN):
                kind = "kan"
            else:
                continue
            feat = candidate_feature(env, p_idx, a, state_feat)
            model, thr = self.heads[kind]
            with torch.no_grad():
                score = float(torch.sigmoid(model(
                    torch.from_numpy(feat).float()[None]))[0])
            if score >= thr and (best is None or score > best[0]):
                best = (score, a)
        return best[1] if best else None


class SuphxFullAgent:
    """完整外部模型：弃牌 Suphx discard 头 + 叫牌 Suphx chi/pon/kan 头 + 规则兜底。"""

    kind = "suphx_full"

    def __init__(self, name="suphxfull"):
        from external_suphx import SuphxAgent
        sys.path.insert(0, ROOT)
        self.base = SuphxAgent(name)
        self.calls = SuphxCallTeacher()
        self.hstate = None
        self.n_calls = 0

    def set_game(self, env, hstate):
        self.hstate = hstate
        if hasattr(self.base, "set_game"):
            self.base.set_game(env, hstate)

    def act(self, env, p_idx, legal):
        from mahjong import env as E
        from external_suphx import build_feature
        # 弃牌阶段：交给 SuphxAgent（会处理自摸/立直/流局 + 弃牌）
        if any(a.type == E.ACT_DISCARD for a in legal):
            return self.base.act(env, p_idx, legal)
        # 响应阶段：荣和优先
        ron = [a for a in legal if a.type == E.ACT_RON]
        if ron:
            return ron[0]
        calls = [a for a in legal if a.type in
                 (E.ACT_PON, E.ACT_CHI, E.ACT_MINKAN, E.ACT_ANKAN, E.ACT_ADD_KAN)]
        if calls:
            state = build_feature(env, p_idx, self.hstate)
            pick = self.calls.decide(env, p_idx, legal, state)
            if pick is not None:
                self.n_calls += 1
                return pick
            return E.PASS_ACTION
        return self.base.act(env, p_idx, legal)
