"""DualStreamMahjongNet 在线 Agent：从 env 实时提特征喂双流 Transformer 选动作。"""
from __future__ import annotations
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mahjong import env as E
from mahjong.tiles import NUM_TILES
from mahjong.dual_net import DualStreamMahjongNet
from mahjong import obs2 as O2
from mahjong import event_stream as ES
from train_hanchan import slot_of, N_ACTIONS


def env_to_features(env, p_idx, encoder):
    """从 MahjongEnv 提 obs2 特征（planes + scalar）。"""
    planes, scalar = encoder.encode(env, p_idx)
    return planes.ravel(), scalar


class DualStreamAgent:
    """DualStreamMahjongNet 在线 Agent：只决定弃牌，响应阶段 pass。"""

    def __init__(self, name, ckpt):
        self.name = name
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model = DualStreamMahjongNet(
            d_model=int(ck.get("d_model", 128)),
            n_layers=int(ck.get("n_layers", 3)),
            n_heads=int(ck.get("n_heads", 4)),
            n_actions=int(ck.get("n_actions", 47)),
            touch_dim=int(ck.get("touch_dim", 8)),
            extra_scalar=int(ck.get("extra_scalar", 12)),
            in_planes=int(ck.get("in_planes", 29)),
            aux_dim=int(ck.get("aux_dim", 0)),
        )
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.encoder = O2.ObsEncoder(order=bool(ck.get("obs_order", False)))
        self.hstate = None

    def set_game(self, env, hstate=None):
        self.hstate = hstate

    def act(self, env, p, legal):
        # 和牌必和
        for wt in (E.ACT_TSUMO, E.ACT_RON):
            ws = [a for a in legal if a.type == wt]
            if ws:
                return ws[0]
        # 响应阶段 pass
        if env.phase == E.Phase.RESPONSE:
            return E.PASS_ACTION if E.PASS_ACTION in legal else legal[0]
        # 弃牌：MLP 选槽
        ds = [a for a in legal if a.type == E.ACT_DISCARD]
        if not ds:
            return E.PASS_ACTION if E.PASS_ACTION in legal else legal[0]
        planes, scalar = self.encoder.encode(env, p)
        obs = np.concatenate([planes.ravel(), scalar]).astype(np.float32)
        tf_feat = np.zeros(8, dtype=np.float32)  # touch features
        ev = ES.encode_event_stream(env, p, max_len=getattr(self.model, "max_event_len", ES.DEFAULT_EVENT_LEN))
        with torch.no_grad():
            logits = self.model(
                torch.from_numpy(obs).float()[None],
                torch.from_numpy(tf_feat).float()[None],
                ev,
            )[0].numpy()
        mask = np.full(N_ACTIONS, -1e9, dtype=np.float32)
        slots = [slot_of(a) for a in ds]
        for s in slots:
            if 0 <= s < N_ACTIONS:
                mask[s] = 0.0
        pick = int((logits + mask).argmax())
        for a, s in zip(ds, slots):
            if s == pick:
                return a
        return ds[0]
