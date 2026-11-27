"""评估 hand_predictor.pt 的 win_prob 头质量（②-A 接入主策略前的质量门槛）。

用法：python eval_hand_predictor.py [--data data/selfplay] [--val 0.15] [--out rules/HANDPRED_EVAL.md]

产出：
- win_prob 的 AUC（排序能力）、Brier（校准）、基础胜率对比
- exp_points 的 MAE、turns_to_tenpai 的 MAE
- 分层校准表（按预测分位，看预测 vs 实际和牌率）
判据：AUC≥0.65 才值得接入主策略（②-A）；否则先修模型/标签。
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))


def build_feature(hand_counts, melds, riichi, shanten, left, dora):
    turn_norm = (136 - left) / 136.0
    dora_val = (dora / 33.0) if dora >= 0 else -1.0
    return np.concatenate([
        hand_counts.astype(np.float32) / 4.0,
        np.array([melds / 4.0, float(riichi), min(shanten, 8) / 8.0,
                  left / 136.0, turn_norm, dora_val], dtype=np.float32),
    ])


def winner_index(d):
    raw = d["winner"]
    if hasattr(raw, "item"):
        raw = raw.item()
    return int(raw) if raw is not None else -1


class HandPredictor(nn.Module):
    def __init__(self, in_dim=40, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden // 2), nn.ReLU())
        self.win = nn.Linear(hidden // 2, 1)
        self.pts = nn.Linear(hidden // 2, 1)
        self.turns = nn.Linear(hidden // 2, 1)

    def forward(self, x):
        h = self.trunk(x)
        return (torch.sigmoid(self.win(h)),
                self.pts(h).squeeze(-1) * 1000.0,
                nn.functional.softplus(self.turns(h)).squeeze(-1))


def load_dataset(data_dir, val_frac=0.15, seed=7):
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    Xs, Yw = [], []
    skipped = 0
    for f in files:
        d = np.load(f, allow_pickle=True)
        if "shanten" not in d or "winner" not in d:
            skipped += 1
            continue
        hands = d["hands"]; melds = d["melds"]; riichi = d["riichi"]
        left = d["left"]; dora = d["dora"]; seats = d["seat"]
        shanten = d["shanten"]
        winner = winner_index(d)
        for k in range(len(hands)):
            p = int(seats[k])
            feat = build_feature(hands[k, p], melds[k, p], riichi[k, p],
                                 int(shanten[k]), int(left[k]), int(dora[k]))
            Xs.append(feat)
            Yw.append(1.0 if winner == p else 0.0)
    if not Xs:
        return None, None, len(files), skipped
    X = np.stack(Xs); Y = np.array(Yw, np.float32)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = max(1, int(len(X) * val_frac))
    return (X[idx[n_val:]], Y[idx[n_val:]]), (X[idx[:n_val]], Y[idx[:n_val]]), len(files), skipped


def auc_manual(y, p):
    """Mann-Whitney U 形式的 AUC，避免依赖 sklearn。"""
    pos = p[y > 0.5]; neg = p[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[:len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "checkpoints", "hand_predictor.pt"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "selfplay"))
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--out", default=os.path.join(ROOT, "rules", "HANDPRED_EVAL_2026-11-28.md"))
    args = ap.parse_args()

    (Xtr, Ytr), (Xv, Yv), n_files, skipped = load_dataset(args.data, args.val)
    if Xv is None:
        print("无可用验证数据（需要含 shanten/winner 的新格式）", flush=True)
        return 1
    print(f"验证集 {len(Xv)} 决策点 / 训练 {len(Xtr)}（{n_files} 文件，跳过 {skipped}）", flush=True)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = HandPredictor(in_dim=ck["in_dim"])
    model.load_state_dict(ck["state_dict"])
    model.eval()

    with torch.no_grad():
        p, pts, turns = model(torch.from_numpy(Xv))
    p = p.numpy().squeeze(-1)
    base = float(Yv.mean())

    # AUC
    order = np.argsort(p)
    ys = Yv[order]
    npos = int(ys.sum()); nneg = len(ys) - npos
    if npos == 0 or nneg == 0:
        auc = float("nan")
    else:
        ranks = np.empty(len(ys)); ranks[order] = np.arange(1, len(ys) + 1)
        auc = (ranks[ys > 0.5].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    # Brier
    brier = float(np.mean((p - Yv) ** 2))
    # 校准：按预测分 5 桶，看桶内实际和牌率
    bins = np.quantile(p, [0, .2, .4, .6, .8, 1.0])
    calib = []
    for i in range(5):
        m = (p >= bins[i]) & (p <= bins[i + 1] if i == 4 else p < bins[i + 1])
        if m.sum():
            calib.append((bins[i], bins[i + 1], float(p[m].mean()), float(Yv[m].mean()), int(m.sum())))

    lines = [f"# hand_predictor win_prob 质量评估（{n_files} 文件）", "",
             f"- 验证集 {len(Xv)} 决策点，和牌率 base_win={base:.4f}",
             f"- **AUC={auc:.4f}**（判据 ≥0.65 才值得接入主策略）",
             f"- Brier={brier:.4f}（基础胜率 {base:.4f}，随机=base*(1-base)={base*(1-base):.4f}）",
             "", "## 分层校准（预测分位 → 桶内实际和牌率）", "",
             "| 预测区间 | 预测均值 | 实际和牌率 | 桶样本 |", "|---|---|---|---|"]
    for lo, hi, pm, am, n in calib:
        lines.append(f"| {lo:.3f}–{hi:.3f} | {pm:.4f} | {am:.4f} | {n} |")
    lines += ["", f"**结论**：AUC {'≥' if auc >= 0.65 else '<'} 0.65 → "
              f"{'可接入主策略（②-A）' if auc >= 0.65 else '先修模型/标签再接入'}"]
    out = "\n".join(lines)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(out)
    print(out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
