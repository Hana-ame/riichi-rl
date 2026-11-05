"""实现+实验：牌河顺序特征 → 对手向听/听牌预测（显式头）。

审阅要求「通过牌河顺序判断他家形状/向听」。本脚本做真实实现与消融实验：
- 数据：跑 N 局 4×greedy，在每个决策点记录观察者 o、目标 t（门前 13 张）：
  - 基础特征 108 维（复用 opponent_model.build_features：牌河 counts/副露/立直/巡目等）
  - **顺序特征 816 维**：四家各自最近 6 张弃牌的位次 one-hot（4×6×34，缺失位补 0）
  - 标签：t 的向听数（ref_shanten 0..6，>6 截断）+ 是否听牌
- 训练两个 MLP：只用 108 维（基线） vs 108+816（加顺序），对比：
  - 向听 MAE（越小越好）
  - 听牌二分类 AUC（越大越好）
  消融结论写入 JSON。

用法：
    python opp_shanten.py --deals 60 --epochs 8 --out rules/OPP_SHANTEN_2026-11-05.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn

from mahjong import env as E
from mahjong import refengine as REF
from mahjong.hanchan import HanchanState
from opponent_model import IN_DIM, N_TILES, build_features

ROOT = os.path.dirname(os.path.abspath(__file__))
ORDER_LEN = 6


def order_planes(rivers):
    """四家最近 ORDER_LEN 张弃牌的位次 one-hot：shape (4*ORDER_LEN, 34)。"""
    out = np.zeros((4 * ORDER_LEN, N_TILES), dtype=np.float32)
    for p in range(4):
        seq = rivers[p][-ORDER_LEN:]
        for pos, tile in enumerate(seq):
            out[p * ORDER_LEN + pos, tile] = 1.0
    return out.ravel()


def shanten_label(counts):
    c = counts.astype(np.int32).tolist()
    sh = int(REF.ref_shanten(c))
    return min(sh, 6), 1.0 if sh == 0 else 0.0


class Head(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        self.sh = nn.Linear(hidden // 2, 7)
        self.tenpai = nn.Linear(hidden // 2, 1)

    def forward(self, x):
        h = self.net(x)
        return self.sh(h), self.tenpai(h).squeeze(-1)


def collect(deals, seed=0, use_order=True):
    X, Ysh, Ytp = [], [], []
    for g in range(deals):
        h = HanchanState(dealer=g % 4)
        env = h.new_env(seed=seed + g * 131)
        from mahjong import policies as POL
        pols = {i: POL.make_policy("greedy_tenpai", seed=seed + g * 97 + i)
                for i in range(4)}
        r = env.start_turn()
        steps = 0
        while r is None and steps < 500:
            steps += 1
            o_candidates = env.pending_players()
            if o_candidates:
                o = o_candidates[0]
                obs_hand = np.asarray(env.players[o].counts(), dtype=np.int8)
                rivers = [list(getattr(env.players[i], "river", [])) for i in range(4)]
                for t in range(4):
                    if t == o:
                        continue
                    tp = env.players[t]
                    if len(tp.melds) != 0:
                        continue
                    counts = np.asarray(tp.counts(), dtype=np.int8)
                    if int(counts.sum()) != 13:
                        continue
                    oth = np.zeros(N_TILES, dtype=np.int8)
                    tgt = np.zeros(N_TILES, dtype=np.int8)
                    for tile in rivers[t]:
                        tgt[tile] += 1
                    for j in range(4):
                        if j in (o, t):
                            continue
                        for tile in rivers[j]:
                            oth[tile] += 1
                    base = build_features(
                        obs_hand, tgt, oth, len(tp.melds),
                        1 if tp.riichi else 0, int(tgt.sum()),
                        len(env.wall) - env.wall_idx, int(env.dora_indicator) if env.dora_indicator is not None else -1)
                    if use_order:
                        f = np.concatenate([base, order_planes(rivers)])
                    else:
                        f = base
                    sh, tp_label = shanten_label(counts)
                    X.append(f); Ysh.append(sh); Ytp.append(tp_label)
            acts = {}
            for p in env.pending_players():
                legal = env.legal_actions(p)
                acts[p] = pols[p].choose(env, p, legal) if legal else E.PASS_ACTION
            r = env.step(acts)
        if (g + 1) % 20 == 0:
            print(f"  {g+1}/{deals} 局 样本 {len(X)}", flush=True)
    return (np.stack(X), np.array(Ysh, dtype=np.int64),
            np.array(Ytp, dtype=np.float32))


def train_eval(X, Ysh, Ytp, seed=0, epochs=8, bs=512, lr=1e-3):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = max(1, int(0.1 * len(X)))
    va, tr = idx[:n_val], idx[n_val:]
    model = Head(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.from_numpy(X[tr]); Sht = torch.from_numpy(Ysh[tr]); Tpt = torch.from_numpy(Ytp[tr])
    Xv = torch.from_numpy(X[va]); Shv = torch.from_numpy(Ysh[va]); Tpv = torch.from_numpy(Ytp[va])
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            b = perm[i:i + bs]
            sh, tp = model(Xt[b])
            loss = nn.functional.cross_entropy(sh, Sht[b]) + \
                nn.functional.binary_cross_entropy_with_logits(tp, Tpt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        sh, tp = model(Xv)
    mae = float((sh.argmax(1) - Shv).abs().float().mean())
    # AUC（手写，无 sklearn 依赖）
    scores = torch.sigmoid(tp).numpy(); labels = Tpv.numpy()
    order = np.argsort(scores)
    ranks = np.empty_like(order); ranks[order] = np.arange(len(scores))
    n_pos = labels.sum(); n_neg = len(labels) - n_pos
    if n_pos > 0 and n_neg > 0:
        auc = (ranks[labels == 1].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)
    else:
        auc = float("nan")
    return mae, float(auc), model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=os.path.join(ROOT, "rules",
                                                 "OPP_SHANTEN_2026-11-05.json"))
    args = ap.parse_args()
    torch.set_num_threads(2)
    res = {}
    for name, use_order in (("baseline_108", False), ("with_order_924", True)):
        print(f"== {name} ==", flush=True)
        X, Ysh, Ytp = collect(args.deals, seed=args.seed, use_order=use_order)
        print(f"样本 {len(X)} 特征 {X.shape[1]} 听牌占比 {Ytp.mean():.3f}", flush=True)
        mae, auc, model = train_eval(X, Ysh, Ytp, seed=args.seed,
                                     epochs=args.epochs)
        res[name] = {"samples": len(X), "dim": int(X.shape[1]),
                     "shanten_mae": round(mae, 3),
                     "tenpai_auc": round(auc, 3)}
        print(f"  -> 向听 MAE={mae:.3f}  听牌 AUC={auc:.3f}", flush=True)
        np.savez_compressed(os.path.join(ROOT, "data", f"oppsh_{name}.npz"),
                            x=X[:2000], ysh=Ysh[:2000], ytp=Ytp[:2000])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    print("写入", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
