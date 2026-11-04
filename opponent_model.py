"""对手手牌推断模型：从「可见信息」预测某个对手每张牌的持有概率。

可见信息（输入，观察者 o 视角、目标 t）：
  - o 自己的手牌 counts(34)          （已知）
  - t 的牌河 counts(34)               （公开）
  - 另外两家的牌河 counts 聚合(34)     （公开）
  - t 的副露数、立直、已打张数         （公开）
  - 巡目、墙余、宝牌                  （公开）
输出：34 维，target 每张牌的持有概率 0..1。

监督标签来自观战服务器记录的真实手牌（上帝视角）。数据越多越准，可反复重训迭代。

用法：
    /mnt/d/WorkPlace/my_rl_project/venv/bin/python opponent_model.py --epochs 40
    /mnt/d/WorkPlace/my_rl_project/venv/bin/python opponent_model.py --data data/selfplay --out checkpoints/opponent_model.pt
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

ROOT = os.path.dirname(os.path.abspath(__file__))
N_TILES = 34
IN_DIM = N_TILES * 3 + 6   # 108


class OpponentNet(nn.Module):
    def __init__(self, in_dim=IN_DIM, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, N_TILES),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))


def build_features(obs_hand, target_river, other_rivers, target_melds,
                   target_riichi, target_discards, left, dora):
    """构造一条输入特征（单样本，numpy）。"""
    turn_norm = (136 - left) / 136.0
    dora_val = (dora / 33.0) if dora >= 0 else -1.0
    meta = np.array([
        target_melds / 4.0,
        float(target_riichi),
        target_discards / 18.0,
        turn_norm,
        left / 136.0,
        dora_val,
    ], dtype=np.float32)
    return np.concatenate([
        obs_hand.astype(np.float32) / 4.0,
        target_river.astype(np.float32) / 4.0,
        other_rivers.astype(np.float32) / 4.0,
        meta,
    ])


def load_samples(data_dir):
    """从 npz 构造 (X, Y)。每个快照产生 3 个 target 样本。"""
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    Xs, Ys = [], []
    skipped = 0
    for f in files:
        d = np.load(f, allow_pickle=True)
        if "hands" not in d:
            skipped += 1
            continue
        hands = d["hands"]       # (N,4,34)
        rivers = d["rivers"]
        melds = d["melds"]
        riichi = d["riichi"]
        left = d["left"]
        dora = d["dora"]
        seats = d["seat"]
        for k in range(len(hands)):
            o = int(seats[k])
            for t in range(4):
                if t == o:
                    continue
                other_rivers = np.zeros(N_TILES, dtype=np.int8)
                for i in range(4):
                    if i != o and i != t:
                        other_rivers += rivers[k, i]
                feat = build_features(
                    hands[k, o], rivers[k, t], other_rivers,
                    melds[k, t], riichi[k, t], int(rivers[k, t].sum()),
                    left[k], int(dora[k]))
                Xs.append(feat)
                Ys.append(hands[k, t].astype(np.float32) / 4.0)
    if not Xs:
        return None, None, len(files), skipped
    return np.stack(Xs), np.stack(Ys), len(files), skipped


def train(args):
    X, Y, n_files, skipped = load_samples(args.data)
    if X is None:
        print(f"没有可训数据：{args.data} 里没有含 hands 的记录"
              f"（旧格式文件 {skipped} 个被跳过）。先跑观战服务器攒几局再训。")
        return 1
    print(f"样本 {len(X)} 条（来自 {n_files} 个文件，跳过旧格式 {skipped} 个）"
          f"  输入 {X.shape[1]} 维", flush=True)

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    n_val = max(1, int(len(X) * args.val))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    Xt = torch.from_numpy(X[tr_idx]); Yt = torch.from_numpy(Y[tr_idx])
    Xv = torch.from_numpy(X[val_idx]); Yv = torch.from_numpy(Y[val_idx])

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = OpponentNet().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"device={dev}  训练 {len(tr_idx)} / 验证 {len(val_idx)}", flush=True)

    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(Xt), args.bs):
            b = perm[i:i + args.bs]
            xb, yb = Xt[b].to(dev), Yt[b].to(dev)
            pred = model(xb)
            loss = nn.functional.binary_cross_entropy(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(b)
        # 验证
        model.eval()
        with torch.no_grad():
            pv = model(Xv.to(dev)).cpu().numpy()
        vloss = float(nn.functional.binary_cross_entropy(
            torch.from_numpy(pv), Yv))
        # top5 命中：预测概率最高的5张里，真实持有的占比
        top5 = np.argsort(-pv, axis=1)[:, :5]
        hit = 0
        for i in range(len(pv)):
            true_held = np.where(Yv[i].numpy() > 0)[0]
            hit += len(set(top5[i]) & set(true_held.tolist())) / max(1, len(true_held))
        acc5 = hit / len(pv)
        print(f"epoch {ep:>2}  train_loss={tot/len(Xt):.4f}  "
              f"val_loss={vloss:.4f}  top5_hit={acc5:.3f}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "in_dim": IN_DIM,
                "n_tiles": N_TILES, "samples": len(X),
                "val_loss": float(vloss)}, args.out)
    print(f"saved {args.out}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "selfplay"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--out",
                    default=os.path.join(ROOT, "checkpoints", "opponent_model.pt"))
    args = ap.parse_args()
    return train(args)


if __name__ == "__main__":
    sys.exit(main())