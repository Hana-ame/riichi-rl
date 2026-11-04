"""对手「和牌张」概率模型：从可见信息预测对手每个牌种是不是危险张。

审阅建议（2026-11-05）：「危险分升级为『对手和牌张概率』再 500+ 局验证」。

与 `opponent_model.py`（预测对手**持有**每张牌）不同，本模型预测：
**对手每张牌中，哪些是他听牌的「和牌张」**（即打出去会被荣和的牌）。

监督标签直接从观战数据构造（上帝视角）：
- 目标家门前（副露=0）且手牌 13 张；
- `ref_shanten(counts) == 0` 时，用 `can_win_with` 枚举全部和牌张 → 34 维二值标签；
- 非听牌 → 全 0。
排除副露玩家（记录里只存副露数、没有副露具体牌，无法还原形状）。

输入特征复用 `opponent_model.build_features`（自家手牌 + 目标家牌河 +
其它两家牌河聚合 + 副露数/立直/已打张数 + 巡目/墙余/宝牌）。

用法：
    python danger_model.py --epochs 10 --out checkpoints/danger_model.pt
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mahjong import refengine as REF
from opponent_model import IN_DIM, N_TILES, build_features

ROOT = os.path.dirname(os.path.abspath(__file__))


class DangerNet(nn.Module):
    """108 维可见信息 -> 34 维「对手和牌张概率」。"""

    def __init__(self, in_dim=IN_DIM, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, N_TILES),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))


def waits_of(counts):
    """13 张门前手牌的全部和牌张（34 维 0/1）。"""
    y = np.zeros(N_TILES, dtype=np.float32)
    c = counts.astype(np.int32).tolist()
    if sum(c) != 13:
        return y, False
    if REF.ref_shanten(c) != 0:
        return y, False
    for t in range(N_TILES):
        if c[t] >= 4:
            continue
        if REF.can_win_with(c, t, 0):
            y[t] = 1.0
    return y, True


def load_samples(data_dir, limit_files=0):
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if limit_files:
        files = files[-limit_files:]
    Xs, Ys = [], []
    skipped = 0
    for f in files:
        d = np.load(f, allow_pickle=True)
        if "hands" not in d:
            skipped += 1
            continue
        hands = d["hands"]; rivers = d["rivers"]; melds = d["melds"]
        riichi = d["riichi"]; left = d["left"]; dora = d["dora"]
        seats = d["seat"]
        for k in range(len(hands)):
            o = int(seats[k])
            for t in range(4):
                if t == o:
                    continue
                if int(melds[k, t]) != 0:
                    continue          # 有副露无法还原形状，跳过
                y, _tenpai = waits_of(hands[k, t])   # 非听牌 -> 全 0 负样本
                other = np.zeros(N_TILES, dtype=np.int8)
                for i in range(4):
                    if i != o and i != t:
                        other += rivers[k, i]
                fvec = build_features(
                    hands[k, o], rivers[k, t], other, int(melds[k, t]),
                    int(riichi[k, t]), int(rivers[k, t].sum()),
                    int(left[k]), int(dora[k]))
                Xs.append(fvec)
                Ys.append(y)
    if not Xs:
        return None, None, len(files), skipped
    return (np.stack(Xs), np.stack(Ys), len(files), skipped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "selfplay"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--out", default=os.path.join(
        ROOT, "checkpoints", "danger_model.pt"))
    args = ap.parse_args()

    X, Y, n_files, skipped = load_samples(args.data)
    if X is None:
        print("没有可训数据。")
        return 1
    print(f"样本 {len(X)}（{n_files} 文件，跳过旧格式 {skipped}）"
          f"  正样本率 {float(Y.mean()):.3f}", flush=True)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    n_val = max(1, int(len(X) * args.val))
    va, tr = idx[:n_val], idx[n_val:]
    Xt = torch.from_numpy(X[tr]); Yt = torch.from_numpy(Y[tr])
    Xv = torch.from_numpy(X[va]); Yv = torch.from_numpy(Y[va])
    # 正样本（听牌时刻的和牌张）极稀疏：BCE pos_weight=负/正 处理不平衡
    n_pos = float(Yt.sum())
    n_neg = float(Yt.numel() - n_pos)
    pos_weight = torch.tensor(max(1.0, n_neg / max(1.0, n_pos)))
    model = DangerNet()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"device=cpu 训练 {len(tr)} / 验证 {len(va)}  pos_weight={float(pos_weight):.1f}",
          flush=True)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(Xt), args.bs):
            b = perm[i:i + args.bs]
            pred = model(Xt[b])
            # predict=sigmoid 输出：用逐元素权重模拟 pos_weight
            wb = 1.0 + (pos_weight - 1.0) * Yt[b]
            loss = nn.functional.binary_cross_entropy(pred, Yt[b], weight=wb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(b)
        model.eval()
        with torch.no_grad():
            pv = model(Xv)
            wv = 1.0 + (pos_weight - 1.0) * Yv
            vloss = float(nn.functional.binary_cross_entropy(
                pv, Yv, weight=wv))
            top5 = np.argsort(-pv.numpy(), axis=1)[:, :5]
            hit = 0
            cnt = 0
            for i in range(len(pv)):
                true = np.where(Yv[i].numpy() > 0)[0]
                if len(true) == 0:
                    continue
                cnt += 1
                hit += len(set(top5[i]) & set(true.tolist())) / len(true)
        print(f"epoch {ep:>2}  train_loss={tot/len(Xt):.4f}  "
              f"val_loss={vloss:.4f}  top5_hit={hit/max(1,cnt):.3f} "
              f"(仅听牌样本 n={cnt})", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "in_dim": IN_DIM,
                "n_tiles": N_TILES, "samples": len(X),
                "val_loss": float(vloss)}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())