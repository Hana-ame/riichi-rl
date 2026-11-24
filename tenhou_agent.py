"""Tenhou MLP 在线 Agent：从 env 实时提 560 维特征喂 MLP 选动作。

训练时用 houou_data.Decision 的 560 维特征（见 tenhou_offline_bc.py），
推理时从 MahjongEnv 直接抽取相同字段。不匹配项（ melds_by 元组 vs melds 列表）
按同一编码规则对齐。

用法：
    python eval_selfplay_gate.py --tenhou-ckpt checkpoints/tenhou_mlp.pt --deals 240
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mahjong import env as E
from mahjong.tiles import NUM_TILES
from train_hanchan import slot_of, N_ACTIONS


def env_to_features(env, p_idx):
    """从 MahjongEnv 提 560 维特征（与 tenhou_offline_bc 决策字段对齐）。"""
    f = np.zeros(560, dtype=np.float32)
    p = env.players[p_idx]

    # 0-33: 自己手牌
    hand = p.hand
    for t in hand:
        if 0 <= t < 34:
            f[t] += 1

    # 34-67: 自己弃牌
    for t in p.river:
        if 0 <= t < 34:
            f[34 + t] += 1

    # 68-203: 四家弃牌（相对视角 0=自家，1=下家，2=对家，3=上家）
    for offset in range(4):
        q = env.players[(p_idx + offset) % 4]
        for t in q.river:
            if 0 <= t < 34:
                f[68 + offset * 34 + t] += 1

    # 204-339: 四家副露
    # 训练时 houou_data.Decision.melds_by 是 [(t1,t2,t3),...]，每个 tile +1
    # 推理时 env.players[*].melds 是 [(kind, tile, consumed),...]，需按同一规则对齐
    for offset in range(4):
        q = env.players[(p_idx + offset) % 4]
        for kind, t, consumed in q.melds:
            if kind == "chi":
                # chi 算连续 3 张：t, t+1, t+2（与训练时 melds_by 元组内 3 个 tile 等价）
                for dt in range(3):
                    if t + dt < 34:
                        f[204 + offset * 34 + t + dt] += 1
            elif kind == "pon":
                # pon 算 3 张相同牌
                if 0 <= t < 34:
                    f[204 + offset * 34 + t] += 3
            elif kind in ("ankan", "minkan", "addkan"):
                # 杠算 4 张
                if 0 <= t < 34:
                    f[204 + offset * 34 + t] += 4

    # 340-509: dora 指示牌
    inds = env.dora_indicators if hasattr(env, "dora_indicators") else []
    if not inds and env.dora_indicator is not None:
        inds = [env.dora_indicator]
    for i, ind in enumerate(inds[:5]):
        if ind is not None and 0 <= ind < 34:
            f[340 + i * 34 + ind] = 1

    # 510-513: 四家立直
    for offset in range(4):
        q = env.players[(p_idx + offset) % 4]
        if q.riichi:
            f[510 + offset] = 1

    # 514-516: 局面
    f[514] = (getattr(env, "kyoku", 1) - 1) / 4.0
    f[515] = min(getattr(env, "honba", 0), 3) / 3.0
    f[516] = max(0, min(70, len(env.wall) - env.wall_idx)) / 70.0

    # 517-520: 手牌/14, 副露/4, 轮次/20, 门清
    f[517] = len(hand) / 14.0
    f[518] = len(p.melds) / 4.0
    f[519] = min(getattr(env, "turn", 0), 20) / 20.0
    f[520] = 1.0 if len(p.melds) == 0 else 0.0

    # 521-523: trigger_type one-hot（弃牌=self=0，响应=call=1）
    if env.phase == E.Phase.RESPONSE:
        f[522] = 1
    else:
        f[521] = 1

    # 524-558: trigger_tile（响应类）
    if env.phase == E.Phase.RESPONSE and env.last_discard is not None:
        f[524] = 1
        if env.last_discard < 34:
            f[525 + env.last_discard] = 1

    return f


class TenhouMLPAgent:
    """Tenhou MLP 在线 Agent：只决定弃牌，响应阶段 pass。"""

    def __init__(self, name, ckpt):
        self.name = name
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        in_dim = int(ck.get("in_dim", 560))
        hidden = int(ck.get("hidden", 512))
        n_actions = int(ck.get("n_actions", 47))
        self.model = _ActionMLP(in_dim, hidden, n_actions)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.hstate = None

    def set_game(self, env, hstate=None):
        self.hstate = hstate

    def act(self, env, p, legal):
        # 和牌必和（规则接管，与训练时 expert 标签一致）
        for wt in (E.ACT_TSUMO, E.ACT_RON):
            ws = [a for a in legal if a.type == wt]
            if ws:
                return ws[0]
        # MLP 在全部合法动作上选（训练标签含 RIICHI/CHI/PON/KAN/PASS）
        f = env_to_features(env, p)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(f).unsqueeze(0))[0].numpy()
        mask = np.full(N_ACTIONS, -1e9, dtype=np.float32)
        pairs = []  # (slot, action)
        for a in legal:
            s = slot_of(a)
            if 0 <= s < N_ACTIONS:
                pairs.append((s, a))
                mask[s] = 0.0
        if not pairs:
            return E.PASS_ACTION if E.PASS_ACTION in legal else legal[0]
        pick = int((logits + mask).argmax())
        for s, a in pairs:
            if s == pick:
                return a
        return legal[0]


class _ActionMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden, n_actions):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden // 2), torch.nn.ReLU(),
            torch.nn.Linear(hidden // 2, n_actions),
        )

    def forward(self, x):
        return self.net(x)
