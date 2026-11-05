"""外部模型（Suphx 复现）适配器：让它的弃牌模型在我们的立直 env 里打牌。

可行性已核实：其 `get_feature()` 实测输出 **(291,34)**，与公开权重
`in_channels=291` 完全一致；本文件用我们的 env 状态复刻同样 291 通道特征，
弃牌交给它的 50 层 ResNet，立直/和了/副露用我们的规则（GreedyTenpai）兜底。

用法：
    from external_suphx import SuphxAgent
    # 或在 version_benchmark.make_agent 里用 "suphx" 前缀
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(ROOT, "external", "Mahjong-AI")
CKPT = os.path.join(EXT, "model", "saved", "discard-model", "best.pt")

NUM_TILES = 34
DISCARD_LEN = 24

UPPER = np.tril(np.ones(4))


def _cat(category, cat_num):
    f = np.zeros((cat_num, NUM_TILES), dtype=np.float32)
    f[category].fill(1.0)
    return f


def _bucket(values, bins):
    cats = np.digitize(values, bins=bins)
    return np.concatenate([_cat(int(c), len(bins) + 1) for c in cats], axis=0)


def _hand_feature(counts):
    f = np.zeros((4, NUM_TILES), dtype=np.float32)
    for t, c in enumerate(counts):
        if c > 0:
            f[:, t] = UPPER[c - 1]
    return f


def _dora_of(indicator_type):
    if indicator_type < 27:
        base = (indicator_type // 9) * 9
        return base + (indicator_type - base + 1) % 9
    return 27 + (indicator_type - 27 + 1) % 4


def _furo_feature(melds):
    feats = []
    for kind, tile, _arg in melds:
        f = np.zeros((4, NUM_TILES), dtype=np.float32)
        if kind == "chi":
            f[0, [tile, tile + 1, tile + 2]] = 1.0
        elif kind == "pon":
            f[[0, 1, 2], tile] = 1.0
        else:  # minkan / ankan
            f[:, tile] = 1.0
        feats.append(f)
    while len(feats) < 4:
        feats.append(np.zeros((4, NUM_TILES), dtype=np.float32))
    return np.concatenate(feats[:4], axis=0)


def build_feature(env, p_idx, hstate):
    """复刻 Suphx 复现的 291×34 特征。"""
    p = env.players[p_idx]
    counts = np.asarray(p.counts(), dtype=np.int32)
    left = len(env.wall) - env.wall_idx
    dora_types = [_dora_of(int(t)) for t in getattr(env, "dora_indicators", [])][:5]
    dora_feat = np.zeros((5, NUM_TILES), dtype=np.float32)
    for i, d in enumerate(dora_types):
        dora_feat[i, d] = 1.0

    # 可见牌 = 四家牌河 + 自家手牌
    visible = np.zeros(NUM_TILES, dtype=np.int32)
    for q in range(4):
        for t in env.players[q].river:
            visible[t] += 1
    for t, c in enumerate(counts):
        visible[t] += c
    vis_feat = np.zeros((4, NUM_TILES), dtype=np.float32)
    for t, c in enumerate(visible):
        if c > 0:
            vis_feat[:, t] = UPPER[min(c, 4) - 1]

    # 舍牌（顺序，4×24）
    disc = np.zeros((4 * DISCARD_LEN, NUM_TILES), dtype=np.float32)
    for q in range(4):
        for j, t in enumerate(env.players[q].river[-DISCARD_LEN:]):
            disc[q * DISCARD_LEN + j, t] = 1.0

    # aka：我们的引擎不区分赤牌，用「手里有 5」近似
    aka = np.zeros((3, NUM_TILES), dtype=np.float32)
    for i, t in enumerate((4, 13, 22)):
        if counts[t] > 0:
            aka[i].fill(1.0)

    # 自风 / 场风（tile type：東27 南28 西29 北30）
    seat_wind = int(p.seat_wind) if 0 <= int(p.seat_wind) < NUM_TILES else 27
    round_wind = 27 if getattr(hstate, "round", "E") == "E" else 28
    wind_feat = np.zeros((2, NUM_TILES), dtype=np.float32)
    wind_feat[0, seat_wind] = 1.0
    wind_feat[1, round_wind] = 1.0

    seat = _cat(p_idx, 4)
    scores = [env.players[q].points for q in range(4)]
    order = sorted(range(4), key=lambda i: (-scores[i], i))
    rank = _cat(order.index(p_idx), 4)
    round_idx = (0 if getattr(hstate, "round", "E") == "E" else 4) + \
        (getattr(hstate, "kyoku", 1) - 1)
    round_feat = _cat(min(15, max(0, round_idx)), 16)
    honba = _cat(min(19, int(getattr(hstate, "honba", 0))), 20)
    riichi_ba = _cat(min(19, int(getattr(hstate, "kyokuten", 0)) // 1000), 20)
    # score bucket: 原始是分数/100，bins=range(50,450,50)（9 档）
    score_feat = _bucket([s / 100.0 for s in scores], list(range(50, 450, 50)))
    reach = np.zeros((4, NUM_TILES), dtype=np.float32)
    for q in range(4):
        reach[q].fill(1.0 if env.players[q].riichi else 0.0)
    oya = _cat(int(getattr(hstate, "dealer", 0)) % 4, 4)
    left_feat = _bucket([left], [5, 10, 22, 46])
    furo = np.concatenate([_furo_feature(env.players[q].melds) for q in range(4)],
                          axis=0)
    return np.concatenate([
        _hand_feature(counts),   # 16
        seat,                    # 4
        rank,                    # 4
        disc,                    # 96
        vis_feat,                # 4
        dora_feat,               # 5
        aka,                     # 3
        wind_feat,               # 2
        left_feat,               # 5
        furo,                    # 64
        round_feat,              # 16
        honba,                   # 20
        riichi_ba,               # 20
        score_feat,              # 36
        reach,                   # 4
        oya,                     # 4
    ], axis=0)


class SuphxAgent:
    """我们的 env Agent 接口；弃牌用 Suphx 复现模型，其余动作用规则。"""

    kind = "suphx_external"

    def __init__(self, name="suphx", ckpt=None):
        from arena_elo import RuleAgent
        self.name = name
        self.hstate = None
        self.rule = RuleAgent("greedy_tenpai")
        ck_path = ckpt or CKPT
        path = os.path.join(EXT, "model", "models.py")
        sys.path.insert(0, EXT)
        from model.models import DiscardModel
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        self.model = DiscardModel(num_layers=ck["num_layers"],
                                  in_channels=ck["in_channels"])
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        sys.path.remove(EXT)

    def set_game(self, env, hstate=None):
        self.hstate = hstate

    def act(self, env, p_idx, legal):
        from mahjong import env as E
        discards = [a for a in legal if a.type == E.ACT_DISCARD]
        if not discards:
            return self.rule.act(env, p_idx, legal)
        # 和了/立直等特殊动作优先交给规则
        specials = [a for a in legal if a.type in
                    (E.ACT_TSUMO, E.ACT_RIICHI, E.ACT_RYUUKYOKU, E.ACT_NAGASHI)]
        if specials:
            pick = self.rule.act(env, p_idx, specials)
            if pick.type != E.ACT_PASS:
                return pick
        feat = build_feature(env, p_idx, self.hstate)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(feat).float()[None])[0].numpy()
        avail = sorted({a.tile for a in discards})
        tile = max(avail, key=lambda t: logits[t])
        return next(a for a in discards if a.tile == tile)
