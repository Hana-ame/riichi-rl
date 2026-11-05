"""外部立直模型 riichippo (zangjiucheng, Apache-2.0, 74ch/192w/16block ResNet, 82动作)
适配到我们的 MahjongEnv：把我们的局面编码成 RiichiEnv 的 74 通道特征，跑其 logits。

特征依据：RiichiEnv docs/FEATURE_ENCODING.md；动作依据 docs/ENCODING.md（82 动作）。
近似：红牌通道=0（我们引擎 34 种牌、无红牌信息）；杠/防守等细节按公开信息近似。
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn
from mahjong import env as E, refengine as REF
from mahjong.env import (ACT_DISCARD, ACT_TSUMO, ACT_RIICHI, ACT_ANKAN, ACT_MINKAN,
                         ACT_CHI, ACT_PON, ACT_ADD_KAN, ACT_RON, ACT_PASS, PASS_ACTION)
from mahjong.tiles import NUM_TILES

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "external", "riichippo", "v1.pth")


class ChannelAttention(nn.Module):
    def __init__(self, ch, ratio=16):
        super().__init__()
        self.shared_mlp = nn.Sequential(nn.Linear(ch, ch // ratio), nn.ReLU(inplace=True),
                                        nn.Linear(ch // ratio, ch))
    def forward(self, x):
        w = (self.shared_mlp(x.mean(-1)) + self.shared_mlp(x.amax(-1))).sigmoid()
        return w.unsqueeze(-1) * x


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=1); self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1); self.bn2 = nn.BatchNorm1d(ch)
        self.ca = ChannelAttention(ch)
    def forward(self, x):
        h = torch.relu(self.bn1(self.conv1(x)))
        h = self.ca(self.bn2(self.conv2(h)))
        return torch.relu(h + x)


class ResNetBackbone(nn.Module):
    def __init__(self, in_ch=74, conv=192, blocks=16, fc=512):
        super().__init__()
        self.conv_in = nn.Conv1d(in_ch, conv, 3, padding=1)
        self.bn_in = nn.BatchNorm1d(conv)
        self.res_blocks = nn.ModuleList([ResBlock(conv) for _ in range(blocks)])
        self.fc = nn.Linear(conv * NUM_TILES, fc)
    def forward(self, x):
        h = torch.relu(self.bn_in(self.conv_in(x)))
        for b in self.res_blocks: h = b(h)
        return torch.relu(self.fc(h.flatten(1)))


class RiichippoNet(nn.Module):
    """键名严格匹配 v1.pth：backbone.* / actor_head.* / critic_head.*"""
    def __init__(self, in_ch=74, conv=192, blocks=16, fc=512, n_act=82):
        super().__init__()
        self.backbone = ResNetBackbone(in_ch, conv, blocks, fc)
        self.actor_head = nn.Linear(fc, n_act)
        self.critic_head = nn.Linear(fc, 1)
    def forward(self, x):
        h = self.backbone(x)
        return self.actor_head(h), self.critic_head(h).squeeze(-1)


def build_features(env, p_idx):
    """74 x 34，按 RiichiEnv FEATURE_ENCODING.md（相对座位：self,shimocha,toimen,kamicha）。"""
    f = np.zeros((74, NUM_TILES), dtype=np.float32)
    p = env.players[p_idx]
    order = [(p_idx + k) % 4 for k in range(4)]
    own = order[0]

    def meld_tiles(m):
        kind, t, _ = m
        if kind == "chi": return [t, t + 1, t + 2]
        n = 4 if kind in ("ankan", "minkan") else 3
        if kind == "addkan": n = 4
        return [t] * n

    # 0-3 hand counts
    for t, c in enumerate(p.counts()):
        for k in range(4):
            if c >= k + 1: f[k, t] = 1.0
    # 4 red tiles: 引擎无红牌信息 → 0（近似）
    # 5-8 self melds (binary per meld slot)
    for i, m in enumerate(p.melds[:4]):
        for t in meld_tiles(m): f[5 + i, t] = 1.0
    # 9 dora indicators
    for ind in getattr(env, "dora_indicators", []) or []:
        f[9, (ind + 1) % NUM_TILES] = 1.0
    # 10-13 self recent 4 discards (0=最新)
    rev = list(reversed(p.river))[:4]
    for k, t in enumerate(rev): f[10 + k, t] = 1.0
    # 14-25 opponents recent 4
    for oi in range(1, 4):
        q = env.players[order[oi]]
        for k, t in enumerate(list(reversed(q.river))[:4]):
            f[14 + (oi - 1) * 4 + k, t] = 1.0
    # 26-29 discard counts /24（相对座位）
    for oi in range(4):
        f[26 + oi, :] = len(env.players[order[oi]].river) / 24.0
    # 30 tiles left /70
    f[30, :] = max(0.0, (len(env.wall) - env.wall_idx) / 70.0)
    # 31 self riichi, 32-34 opp riichi
    f[31, :] = 1.0 if p.riichi else 0.0
    for oi in range(1, 4):
        f[31 + oi, :] = 1.0 if env.players[order[oi]].riichi else 0.0
    # 35 round wind, 36 self wind
    f[35, env.round_wind] = 1.0
    f[36, p.seat_wind] = 1.0
    # 37 honba/10, 38 riichi sticks/5（引擎无 riichi_sticks → 0）
    f[37, :] = getattr(env, "honba", 0) / 10.0
    # 39-42 scores/100000, 43-46 scores/30000（相对座位）
    for oi in range(4):
        pts = env.players[order[oi]].points
        f[39 + oi, :] = pts / 100000.0
        f[43 + oi, :] = pts / 30000.0
    # 47 waits / 48 tenpai
    c = p.full_counts(); mc = len(p.melds)
    waits = []
    if sum(c) == 13:
        for t in range(NUM_TILES):
            if c[t] >= 4: continue
            if REF.can_win_with(c, t, mc):
                waits.append(t)
    if waits:
        f[47, waits] = 1.0
        f[48, :] = 1.0
    # 49-52 rank one-hot（按点数降序）
    pts = [env.players[order[i]].points for i in range(4)]
    rank = sorted(range(4), key=lambda i: -pts[i])
    f[49 + rank.index(0), :] = 1.0
    # 53 kyoku/8, 54 round progress
    f[53, :] = getattr(env, "kyoku", 0) / 8.0
    rw = (env.round_wind - 27) if env.round_wind is not None else 0
    f[54, :] = (rw * 4 + getattr(env, "kyoku", 0)) / 7.0
    # 55-58 dora count per player /12（可见：副露+牌河；自家加手牌）
    dora_set = {((ind + 1) % NUM_TILES) for ind in (getattr(env, "dora_indicators", []) or [])}
    for oi in range(4):
        q = env.players[order[oi]]
        cnt = sum(1 for m in q.melds for t in meld_tiles(m) if t in dora_set)
        cnt += sum(1 for t in q.river if t in dora_set)
        if oi == 0:
            cnt += sum(1 for t in p.hand if t in dora_set)
        f[55 + oi, :] = min(1.0, cnt / 12.0)
    # 59-62 meld counts /4
    for oi in range(4):
        f[59 + oi, :] = len(env.players[order[oi]].melds) / 4.0
    # 63 seen tiles /4（自家手牌+全部副露+牌河+指示牌）
    seen = np.zeros(NUM_TILES, dtype=np.float32)
    for q in env.players:
        for t in q.hand: seen[t] += 1
        for t in q.river: seen[t] += 1
        for m in q.melds:
            for t in meld_tiles(m): seen[t] += 1
    for ind in (getattr(env, "dora_indicators", []) or []): seen[ind] += 1
    f[63, :] = np.clip(seen / 4.0, 0, 1)
    # 64-67 self discards 5-8
    rev8 = list(reversed(p.river))
    for k in range(4, 8):
        if k < len(rev8): f[64 + (k - 4), rev8[k]] = 1.0
    # 68-69 opponent1 (shimocha) discards 5-6
    rev_o = list(reversed(env.players[order[1]].river))
    for k in range(4, 6):
        if k < len(rev_o): f[68 + (k - 4), rev_o[k]] = 1.0
    # 70-73 tsumogiri flags：引擎不记录 → 0（近似）
    return f


class RiichippoAgent:
    """外部 riichippo 策略：弃牌/立直/和了/吃碰杠全由模型 82 动作头决定。"""
    def __init__(self, name="riichippo", ckpt=CKPT, device="cpu"):
        self.name = name
        self.net = RiichippoNet()
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        self.net.load_state_dict(sd)
        self.net.eval()
        self.hstate = None
    def set_game(self, env, hstate):
        self.hstate = hstate
    def act(self, env, p, legal):
        f = torch.from_numpy(build_features(env, p))[None]
        with torch.no_grad():
            logits = self.net(f)[0][0].numpy()
        ids = self.action_ids(env, p, legal)
        if not ids:
            return PASS_ACTION
        best = max(ids, key=lambda a: float(logits[a]))
        return self.id_to_action(env, p, legal, best)
    def action_ids(self, env, p, legal):
        out = []
        for a in legal:
            i = self.action_to_id(env, p, a)
            if i is not None: out.append(i)
        return out
    def action_to_id(self, env, p, a):
        t = a.type
        if t == ACT_DISCARD: return int(a.tile) % 34
        if t == ACT_RIICHI: return 37
        if t == ACT_CHI:
            # a.tile = 被叫的牌, a.arg = 顺子起点
            seq = [a.arg, a.arg + 1, a.arg + 2]
            pos = seq.index(a.tile) if a.tile in seq else 1
            if seq[0] >= 27: return 39  # 字牌不能吃，理论不会发生
            return 38 + pos
        if t == ACT_PON: return 41
        if t in (ACT_ANKAN, ACT_MINKAN, ACT_ADD_KAN): return 42 + int(a.tile) % 34
        if t in (ACT_TSUMO, ACT_RON): return 79
        if t == E.ACT_RYUUKYOKU: return 80
        if t == ACT_PASS: return 81
        return None
    def id_to_action(self, env, p, legal, aid):
        for a in legal:
            if self.action_to_id(env, p, a) == aid: return a
        return legal[0]
