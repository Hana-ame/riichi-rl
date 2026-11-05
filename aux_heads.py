"""Stage 1：三个预测辅助头（对手向听 / 对手听牌 / 对手最终和牌概率）。

特征：108 维可见信息（opponent_model.build_features：自家手牌 + 目标家牌河
counts + 其它两家牌河 + 副露/立直/已打/巡目/墙余/宝牌）。**暂无牌河顺序**
（顺序平面在 obs v3 里加，见 opp_shanten.py 的原型指标）。

标签（同一局 replay，上帝视角）：
- opp_shanten：目标家门前手牌 ref_shanten（0..6，副露家跳过）
- opp_tenpai：shanten==0
- win：该目标家是本局和牌者（流局=0）

用法：
    python aux_heads.py --train --epochs 8
    python aux_heads.py --eval
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mahjong import refengine as REF
from opponent_model import IN_DIM, N_TILES, build_features

ROOT = os.path.dirname(os.path.abspath(__file__))


class AuxHeads(nn.Module):
    def __init__(self, in_dim=IN_DIM, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden // 2), nn.ReLU())
        self.sh = nn.Linear(hidden // 2, 7)
        self.tenpai = nn.Linear(hidden // 2, 1)
        self.win = nn.Linear(hidden // 2, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.sh(h), self.tenpai(h).squeeze(-1), self.win(h).squeeze(-1)


def build_dataset(limit_files=0):
    files = sorted(glob.glob(os.path.join(ROOT, "data", "selfplay", "*.npz")))
    if limit_files:
        files = files[-limit_files:]
    X, Ysh, Ytp, Ywin, groups = [], [], [], [], []
    gid = 0
    for f in files:
        d = np.load(f, allow_pickle=True)
        if "hands" not in d:
            continue
        hands, rivers, melds = d["hands"], d["rivers"], d["melds"]
        riichi, left, dora, seats = d["riichi"], d["left"], d["dora"], d["seat"]
        winner = d["winner"].item() if getattr(d["winner"], "shape", ()) == () else d["winner"]
        this_gid = gid
        for k in range(len(hands)):
            o = int(seats[k])
            for t in range(4):
                if t == o or int(melds[k, t]) != 0:
                    continue
                counts = hands[k, t].astype(np.int32)
                if int(counts.sum()) != 13:
                    continue
                other = np.zeros(N_TILES, dtype=np.int8)
                for i in range(4):
                    if i not in (o, t):
                        other += rivers[k, i]
                feat = build_features(
                    hands[k, o], rivers[k, t], other, int(melds[k, t]),
                    int(riichi[k, t]), int(rivers[k, t].sum()),
                    int(left[k]), int(dora[k]))
                sh = min(int(REF.ref_shanten(counts.tolist())), 6)
                X.append(feat); Ysh.append(sh); Ytp.append(1.0 if sh == 0 else 0.0)
                Ywin.append(1.0 if (winner is not None and int(winner) == t) else 0.0)
                groups.append(this_gid)
        gid += 1
    return (np.stack(X), np.array(Ysh, dtype=np.int64),
            np.array(Ytp, dtype=np.float32), np.array(Ywin, dtype=np.float32),
            np.array(groups, dtype=np.int64))


def train(epochs=8, out=None, limit_files=0):
    X, Ysh, Ytp, Ywin, groups = build_dataset(limit_files)
    print(f"样本 {len(X)} 维 {X.shape[1]}  局数 {len(np.unique(groups))}  "
          f"听牌率 {Ytp.mean():.3f}  和牌率 {Ywin.mean():.3f}（按局切分）", flush=True)
    torch.set_num_threads(2)
    # 按「局」切分，避免同一局的多快照同时进 train/val 造成泄漏
    rng = np.random.default_rng(0)
    uniq = np.unique(groups)
    n_val_g = max(1, int(0.1 * len(uniq)))
    val_g = set(rng.choice(uniq, size=n_val_g, replace=False).tolist())
    va = np.array([i for i in range(len(X)) if int(groups[i]) in val_g])
    tr = np.array([i for i in range(len(X)) if int(groups[i]) not in val_g])
    net = AuxHeads(X.shape[1])
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    Xt = torch.from_numpy(X[tr]); Sht = torch.from_numpy(Ysh[tr])
    Tpt = torch.from_numpy(Ytp[tr]); Wt = torch.from_numpy(Ywin[tr])
    Xv = torch.from_numpy(X[va]); Shv = torch.from_numpy(Ysh[va])
    Tpv = torch.from_numpy(Ytp[va]); Wv = torch.from_numpy(Ywin[va])
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 512):
            b = perm[i:i + 512]
            sh, tp, wn = net(Xt[b])
            loss = (nn.functional.cross_entropy(sh, Sht[b])
                    + nn.functional.binary_cross_entropy_with_logits(tp, Tpt[b])
                    + 0.5 * nn.functional.binary_cross_entropy_with_logits(wn, Wt[b]))
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        sh, tp, wn = net(Xv)
    mae = float((sh.argmax(1) - Shv).abs().float().mean())
    auc_t = _auc(torch.sigmoid(tp).numpy(), Tpv.numpy())
    auc_w = _auc(torch.sigmoid(wn).numpy(), Wv.numpy())
    res = {"samples": len(X), "dim": int(X.shape[1]),
           "opp_shanten_mae": round(mae, 3),
           "opp_tenpai_auc": round(auc_t, 3),
           "win_auc": round(auc_w, 3)}
    print(json.dumps(res, ensure_ascii=False), flush=True)
    if out:
        torch.save({"state_dict": net.state_dict(), "in_dim": int(X.shape[1])}, out)
    with open(os.path.join(ROOT, "rules", "AUX_HEADS_2026-11-05.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    return res


def _auc(scores, labels):
    order = np.argsort(scores); ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores))
    n_pos = labels.sum(); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--limit-files", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "checkpoints", "aux_heads.pt"))
    args = ap.parse_args()
    if args.train:
        train(args.epochs, args.out, args.limit_files)
        return 0
    print("用法 --train")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
