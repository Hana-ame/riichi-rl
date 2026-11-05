"""和牌率预测（AUC≥0.6 目标）：顺序 GRU vs MLP，标签=目标家本局最终和牌。

数据：live 对局，逐决策记录 108维可见统计 + 四家弃牌顺序（各24张）；
标签：本局结束时 winner==target（流局=0）。
"""
from __future__ import annotations
import json, os, sys
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mahjong import env as E, refengine as REF
from mahjong.hanchan import HanchanState
from mahjong import policies as POL
from opponent_model import IN_DIM, N_TILES, build_features
ROOT = os.path.dirname(os.path.abspath(__file__))
SEQ = 24; PAD = 34


def collect(deals=40, seed=2026):
    Xs, Ss, Ys, Gs = [], [], [], []
    for g in range(deals):
        h = HanchanState(dealer=g % 4); env = h.new_env(seed=seed + g * 131)
        pols = {i: POL.make_policy("greedy_tenpai", seed=seed + g * 97 + i) for i in range(4)}
        rows = []
        r = env.start_turn(); steps = 0
        while r is None and steps < 500:
            steps += 1
            pend = env.pending_players()
            if pend:
                o = pend[0]
                hand = np.asarray(env.players[o].counts(), dtype=np.int8)
                rivers = [list(env.players[i].river) for i in range(4)]
                for t in range(4):
                    if t == o or len(env.players[t].melds) != 0: continue
                    counts = np.asarray(env.players[t].counts(), dtype=np.int8)
                    if int(counts.sum()) != 13: continue
                    tgt = np.zeros(N_TILES, dtype=np.int8); oth = np.zeros(N_TILES, dtype=np.int8)
                    for x in rivers[t]: tgt[x] += 1
                    for j in range(4):
                        if j in (o, t): continue
                        for x in rivers[j]: oth[x] += 1
                    base = build_features(hand, tgt, oth, len(env.players[t].melds),
                                          1 if env.players[t].riichi else 0, int(tgt.sum()),
                                          len(env.wall) - env.wall_idx,
                                          int(env.dora_indicator) if env.dora_indicator is not None else -1)
                    seq = np.full((4, SEQ), PAD, dtype=np.int64)
                    for p in range(4):
                        s = rivers[p][-SEQ:]
                        seq[p, SEQ - len(s):] = s
                    rows.append((base, seq.ravel(), t))
            acts = {}
            for p in env.pending_players():
                legal = env.legal_actions(p)
                acts[p] = pols[p].choose(env, p, legal) if legal else E.PASS_ACTION
            r = env.step(acts)
        winner = r.winner if r is not None and r.result_type != "ryuukyoku" else None
        for base, seq, t in rows:
            Xs.append(base); Ss.append(seq); Ys.append(1.0 if winner == t else 0.0)
            Gs.append(g)
    return (np.stack(Xs), np.stack(Ss), np.array(Ys, dtype=np.float32),
            np.array(Gs, dtype=np.int64))


class MLP(nn.Module):
    def __init__(self, din, h=256):
        super().__init__(); self.net = nn.Sequential(nn.Linear(din, h), nn.ReLU(), nn.Linear(h, h // 2), nn.ReLU()); self.w = nn.Linear(h // 2, 1)
    def forward(self, x): return self.w(self.net(x)).squeeze(-1)


class GRU(nn.Module):
    def __init__(self, din, h=128):
        super().__init__(); self.emb = nn.Embedding(N_TILES + 1, 16); self.gru = nn.GRU(16, h, batch_first=True)
        self.net = nn.Sequential(nn.Linear(din + h, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU()); self.w = nn.Linear(128, 1)
    def forward(self, x, s):
        _, hstate = self.gru(self.emb(s)); z = self.net(torch.cat([x, hstate.squeeze(0)], 1)); return self.w(z).squeeze(-1)


def auc(scores, labels):
    order = np.argsort(scores); ranks = np.empty_like(order); ranks[order] = np.arange(len(scores))
    npos = labels.sum(); nneg = len(labels) - npos
    if npos == 0 or nneg == 0: return float("nan")
    return float((ranks[labels == 1].sum() - npos * (npos - 1) / 2) / (npos * nneg))


def run(deals=40, epochs=10):
    torch.set_num_threads(2)
    X, S, Y, G = collect(deals)
    print(f"样本 {len(X)} 和牌率 {Y.mean():.3f}", flush=True)
    # 按局切分，避免同局快照泄漏（随机切分曾把 AUC 虚高到 0.998）
    rng = np.random.default_rng(0); uniq = np.unique(G)
    val_g = set(rng.choice(uniq, size=max(1, int(0.1 * len(uniq))), replace=False).tolist())
    va = np.array([i for i in range(len(X)) if int(G[i]) in val_g])
    tr = np.array([i for i in range(len(X)) if int(G[i]) not in val_g])
    res = {}
    pos_w = torch.tensor((1 - Y[tr].mean()) / max(1e-6, Y[tr].mean()))
    for name, use_seq in (("mlp_base108", False), ("gru_order", True)):
        net = GRU(IN_DIM) if use_seq else MLP(IN_DIM)
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        Xt = torch.from_numpy(X[tr]); St = torch.from_numpy(S[tr]); Yt = torch.from_numpy(Y[tr])
        Xv = torch.from_numpy(X[va]); Sv = torch.from_numpy(S[va]); Yv = torch.from_numpy(Y[va])
        for ep in range(epochs):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 512):
                b = perm[i:i + 512]
                p = net(Xt[b], St[b]) if use_seq else net(Xt[b])
                loss = nn.functional.binary_cross_entropy_with_logits(p, Yt[b], pos_weight=pos_w)
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(Xv, Sv) if use_seq else net(Xv)).numpy()
        res[name] = round(auc(pv, Yv.numpy()), 3)
        print(name, "win_auc", res[name], flush=True)
    with open(os.path.join(ROOT, "rules", "WIN_HEAD_SEQ_2026-11-05.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--deals", type=int, default=40); ap.add_argument("--epochs", type=int, default=10)
    a = ap.parse_args(); run(a.deals, a.epochs)
