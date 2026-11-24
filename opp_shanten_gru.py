"""① 牌河顺序→对手向听：GRU 序列模型 vs MLP 基线（真实顺序数据）。

数据：跑 live 对局，记录观察者/目标与**逐张弃牌顺序**；
特征：108 维可见统计 + 四家各 24 张弃牌序列（tile type 0..33，pad=34）；
模型：tile embedding(16) → GRU(64) → 与 108 维拼接 → 向听分类(7)/听牌二分类；
对照：只用 108 维的 MLP（与之前 opp_shanten 基线同标签）。
验收：顺序 GRU 相对无顺序 MLP 的 MAE 降幅 ≥0.2 即算「顺序有增益」。

用法：python opp_shanten_gru.py --deals 40 --epochs 10
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mahjong import env as E, refengine as REF
from mahjong.hanchan import HanchanState
from mahjong import policies as POL
from opponent_model import IN_DIM, N_TILES, build_features
ROOT = os.path.dirname(os.path.abspath(__file__))
SEQ = 48
PAD = 34


def collect(deals=40, seed=2026):
    Xs, seqs, Ysh, Ytp = [], [], [], []
    for g in range(deals):
        h = HanchanState(dealer=g % 4); env = h.new_env(seed=seed + g * 131)
        pols = {i: POL.make_policy("eff_wuke", seed=seed + g * 97 + i) for i in range(4)}
        r = env.start_turn(); steps = 0
        while r is None and steps < 500:
            steps += 1
            pend = env.pending_players()
            if pend:
                o = pend[0]
                hand = np.asarray(env.players[o].counts(), dtype=np.int8)
                rivers = [list(env.players[i].river) for i in range(4)]
                for t in range(4):
                    if t == o or len(env.players[t].melds) != 0:
                        continue
                    counts = np.asarray(env.players[t].counts(), dtype=np.int8)
                    if int(counts.sum()) != 13:
                        continue
                    tgt = np.zeros(N_TILES, dtype=np.int8); oth = np.zeros(N_TILES, dtype=np.int8)
                    for x in rivers[t]: tgt[x] += 1
                    for j in range(4):
                        if j in (o, t): continue
                        for x in rivers[j]: oth[x] += 1
                    base = build_features(hand, tgt, oth, len(env.players[t].melds),
                                          1 if env.players[t].riichi else 0,
                                          int(tgt.sum()), len(env.wall) - env.wall_idx,
                                          int(env.dora_indicator) if env.dora_indicator is not None else -1)
                    seq = np.full((4, SEQ), PAD, dtype=np.int64)
                    for p in range(4):
                        s = rivers[p][-SEQ:]
                        seq[p, SEQ - len(s):] = s
                    sh = min(int(REF.ref_shanten(counts.tolist())), 6)
                    Xs.append(base); seqs.append(seq.ravel())
                    Ysh.append(sh); Ytp.append(1.0 if sh == 0 else 0.0)
            acts = {}
            for p in env.pending_players():
                legal = env.legal_actions(p)
                acts[p] = pols[p].choose(env, p, legal) if legal else E.PASS_ACTION
            r = env.step(acts)
    return (np.stack(Xs), np.stack(seqs), np.array(Ysh, dtype=np.int64),
            np.array(Ytp, dtype=np.float32))


class MLP(nn.Module):
    def __init__(self, din, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, hidden), nn.ReLU(), nn.Linear(hidden, hidden // 2), nn.ReLU())
        self.sh = nn.Linear(hidden // 2, 7); self.tp = nn.Linear(hidden // 2, 1)
    def forward(self, x):
        h = self.net(x); return self.sh(h), self.tp(h).squeeze(-1)


class GRUNet(nn.Module):
    def __init__(self, din, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(N_TILES + 1, 16)
        self.gru = nn.GRU(16, hidden, batch_first=True)
        self.net = nn.Sequential(nn.Linear(din + hidden, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.sh = nn.Linear(128, 7); self.tp = nn.Linear(128, 1)
    def forward(self, x, seq):
        e = self.emb(seq)
        _, h = self.gru(e)
        h = h.squeeze(0)
        z = self.net(torch.cat([x, h], dim=1))
        return self.sh(z), self.tp(z).squeeze(-1)


def auc(scores, labels):
    order = np.argsort(scores); ranks = np.empty_like(order); ranks[order] = np.arange(len(scores))
    npos = labels.sum(); nneg = len(labels) - npos
    if npos == 0 or nneg == 0: return float("nan")
    return float((ranks[labels == 1].sum() - npos * (npos - 1) / 2) / (npos * nneg))


def run(deals=40, epochs=10):
    torch.set_num_threads(2)
    X, S, Ysh, Ytp = collect(deals)
    print(f"样本 {len(X)} 序列 {S.shape} 听牌率 {Ytp.mean():.3f}", flush=True)
    rng = np.random.default_rng(0); idx = rng.permutation(len(X)); nv = max(1, int(0.1 * len(X)))
    va, tr = idx[:nv], idx[nv:]
    res = {}
    Xt = torch.from_numpy(X[tr]); St = torch.from_numpy(S[tr]); Sht = torch.from_numpy(Ysh[tr]); Tpt = torch.from_numpy(Ytp[tr])
    Xv = torch.from_numpy(X[va]); Sv = torch.from_numpy(S[va]); Shv = torch.from_numpy(Ysh[va]); Tpv = torch.from_numpy(Ytp[va])
    for name, build, use_seq in (("mlp_base108", lambda d: MLP(d), False),
                                  ("gru_order", lambda d: GRUNet(d), True)):
        net = build(IN_DIM); opt = torch.optim.Adam(net.parameters(), 1e-3)
        for ep in range(epochs):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 512):
                b = perm[i:i + 512]
                if use_seq:
                    sh, tp = net(Xt[b], St[b])
                else:
                    sh, tp = net(Xt[b])
                loss = nn.functional.cross_entropy(sh, Sht[b]) + nn.functional.binary_cross_entropy_with_logits(tp, Tpt[b])
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            if use_seq: sh, tp = net(Xv, Sv)
            else: sh, tp = net(Xv)
        mae = float((sh.argmax(1) - Shv).abs().float().mean())
        auc_t = auc(torch.sigmoid(tp).mul(0).add(0).numpy(), Tpv.numpy()) if False else auc(torch.sigmoid(tp).numpy(), Tpv.numpy())
        res[name] = {"mae": round(mae, 3), "tenpai_auc": round(auc_t, 3)}
        print(name, res[name], flush=True)
    gain = res["mlp_base108"]["mae"] - res["gru_order"]["mae"]
    res["mae_gain_order_vs_base"] = round(gain, 3)
    print(f"顺序 GRU 相对无顺序 MLP：MAE 降低 {gain:.3f}（验收线 0.2）", flush=True)
    with open(os.path.join(ROOT, "rules", "OPP_SHANTEN_GRU_2026-11-05.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=40); ap.add_argument("--epochs", type=int, default=10)
    a = ap.parse_args(); run(a.deals, a.epochs)
