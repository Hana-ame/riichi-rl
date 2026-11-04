"""和牌预测器：从当前手牌局面预测「最终能胡多大」「还要几巡能听牌」。

三个输出头（共享主干）：
  - win_prob   这局最终和了的概率
  - exp_points 若和了，预期点数（回归，单位 1000 点）
  - turns_to_tenpai 还要多少巡能听牌（0=已听牌，18=这局基本听不了）

标签来自观战记录：同一 npz 是完整一局（含每步向听数、终局结果），
可算「未来第一个向听=0 的点」和「该玩家最终是否和了/拿多少点」。

用法：
    /mnt/d/WorkPlace/my_rl_project/venv/bin/python hand_predictor.py --epochs 40
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
IN_DIM = N_TILES + 6   # 40


class HandPredictor(nn.Module):
    def __init__(self, in_dim=IN_DIM, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        self.win = nn.Linear(hidden // 2, 1)
        self.pts = nn.Linear(hidden // 2, 1)
        self.turns = nn.Linear(hidden // 2, 1)

    def forward(self, x):
        h = self.trunk(x)
        return (torch.sigmoid(self.win(h)),
                self.pts(h).squeeze(-1) * 1000.0,     # 点数
                nn.functional.softplus(self.turns(h)).squeeze(-1))  # 巡数


def build_feature(hand_counts, melds, riichi, shanten, left, dora):
    turn_norm = (136 - left) / 136.0
    dora_val = (dora / 33.0) if dora >= 0 else -1.0
    return np.concatenate([
        hand_counts.astype(np.float32) / 4.0,
        np.array([melds / 4.0, float(riichi), min(shanten, 8) / 8.0,
                  left / 136.0, turn_norm, dora_val], dtype=np.float32),
    ])


def winner_index(d):
    """从单局 npz 读 winner；流局记录里存的是 None → 返回 -1。

    2026-11-05 修复：旧代码直接 `int(d["winner"])`，遇到流局（None）
    抛 TypeError，hand_predictor 训练必崩（此前 `--epochs 20` 的 exit 1
    就是这个）。
    """
    raw = d["winner"]
    if hasattr(raw, "item"):
        raw = raw.item()
    return int(raw) if raw is not None else -1


def load_dataset(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    Xs, Yw, Yp, Yt = [], [], [], []
    skipped = 0
    for f in files:
        d = np.load(f, allow_pickle=True)
        if "shanten" not in d:
            skipped += 1
            continue
        hands = d["hands"]; melds = d["melds"]; riichi = d["riichi"]
        left = d["left"]; dora = d["dora"]; seats = d["seat"]
        shanten = d["shanten"]
        winner = winner_index(d); pdelta = d["points_delta"]
        N = len(hands)
        for k in range(N):
            p = int(seats[k])
            # 未来第一个向听=0 的 p 决策点
            tgt_turns = 18.0
            if shanten[k] <= 0:
                tgt_turns = 0.0
            else:
                for j in range(k + 1, N):
                    if int(seats[j]) == p and shanten[j] <= 0:
                        # 用墙余差估算巡数：left 差 / 4
                        tgt_turns = max(0.0, (left[k] - left[j]) / 4.0)
                        break
            feat = build_feature(hands[k, p], melds[k, p], riichi[k, p],
                                 int(shanten[k]), int(left[k]), int(dora[k]))
            Xs.append(feat)
            Yw.append(1.0 if winner == p else 0.0)
            Yp.append(float(pdelta[p]) / 1000.0 if winner == p else 0.0)
            Yt.append(tgt_turns / 18.0)
    if not Xs:
        return None, None, None, None, len(files), skipped
    return (np.stack(Xs), np.array(Yw, np.float32), np.array(Yp, np.float32),
            np.array(Yt, np.float32), len(files), skipped)


def train(args):
    X, Yw, Yp, Yt, n_files, skipped = load_dataset(args.data)
    if X is None:
        print(f"没有可训数据（需要含 shanten 字段的新格式记录）。"
              f"文件 {n_files}，跳过 {skipped}。先让观战服务器攒几局。")
        return 1
    print(f"样本 {len(X)}（{n_files} 文件，跳旧格式 {skipped}）", flush=True)

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    n_val = max(1, int(len(X) * args.val))
    val, tr = idx[:n_val], idx[n_val:]

    def to_t(a, i): return torch.from_numpy(a[i])
    Xt = to_t(X, tr); Wt = to_t(Yw, tr); Pt = to_t(Yp, tr); Tt = to_t(Yt, tr)
    Xv = to_t(X, val); Wv = to_t(Yw, val); Pv = to_t(Yp, val); Tv = to_t(Yt, val)
    base_win = float(Yw.mean())
    print(f"基础胜率(全预测1)={base_win:.3f}  "
          f"听牌目标均值={float(Yt.mean()):.3f}", flush=True)

    # 项目硬约束：禁止 GPU（cpu_only.py）。旧代码用 cuda-if-available，
    # 既违反约束，验证时 target 没搬设备还会 cuda/cpu 混用报错。
    dev = "cpu"
    model = HandPredictor().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"device={dev} 训练{len(tr)} 验证{len(val)}", flush=True)

    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        tot_w = tot_p = tot_t = 0.0
        for i in range(0, len(Xt), args.bs):
            b = perm[i:i + args.bs]
            pw, pp, pt = model(Xt[b].to(dev))
            lw = nn.functional.binary_cross_entropy(pw.squeeze(-1), Wt[b].to(dev))
            lp = nn.functional.mse_loss(pp, Pt[b].to(dev))
            lt = nn.functional.mse_loss(pt, Tt[b].to(dev))
            loss = lw + 0.5 * lp + 0.5 * lt
            opt.zero_grad(); loss.backward(); opt.step()
            tot_w += float(lw) * len(b); tot_p += float(lp) * len(b)
            tot_t += float(lt) * len(b)
        model.eval()
        with torch.no_grad():
            vw, vp, vt = model(Xv.to(dev))
            vwl = float(nn.functional.binary_cross_entropy(
                vw.squeeze(-1), Wv.to(dev)))
            vpl = float(nn.functional.mse_loss(vp * 1000,
                                               Pv.to(dev) * 1000))  # 点数 MAE
            vtl = float(nn.functional.mse_loss(vt, Tv.to(dev) * 18))  # 巡 MAE
        print(f"epoch {ep:>2}  win_loss={tot_w/len(Xt):.4f}  "
              f"val_win={vwl:.4f}  val_pts_mae={vpl ** 0.5:.1f}点  "
              f"val_turns_mae={vtl ** 0.5:.2f}巡", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "in_dim": IN_DIM,
                "samples": len(X), "base_win": base_win}, args.out)
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
                    default=os.path.join(ROOT, "checkpoints", "hand_predictor.pt"))
    args = ap.parse_args()
    return train(args)


if __name__ == "__main__":
    sys.exit(main())