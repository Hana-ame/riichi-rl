"""诊断：牌河顺序→向听预测 是否存在标签泄漏（2026-11-26）。

背景：rules/OPP_SHANTEN_2026-11-05.md 里 baseline_108 MAE=0.328 / AUC=1.000，
      +顺序 924 MAE=0.310 / AUC=1.000（增益 0.018，AUC 异常偏高）。

怀疑：build_features 的 meta 6 维含 target_riichi / target_discards / turn_norm / left，
      这些与「向听标签」强相关（巡目越大→剩余牌越少→向听必然下降；greedy_tenpai 立直
      ⟺ 听牌），模型可走捷径而非真正从牌河顺序推断。

本脚本做 4 件事（一次采集原始分量，避免重复仿真）：
  1. 相关性分析：108 维每个特征 vs 向听标签 / 听牌标签 的 Pearson r（取 |r| 排序），
     并单列 meta 6 维的 r；再算 巡目×平均向听、立直×听牌 的列联表。
  2. 捷径模型：只用 meta 6 维（无任何牌河/手牌）预测向听/听牌 → 若 MAE 接近 0.31、
     AUC 接近 1.0，即为泄漏铁证。
  3. 控制巡目实验：
     (a) 去掉全部 meta 6 维（no_meta）：baseline_102 vs +顺序 → 牌河顺序的「真实」增益
     (b) 只去 turn/left/discards，保留 riichi（no_time）：baseline_105 vs +顺序
     (c) 同巡目分层评估：full 特征下 baseline vs +顺序 在每个巡目桶内的 MAE 差
  4. 扩数据集：--deals 200 复核稳定性。

用法：
    /home/lumin/miniconda3/bin/python opp_shanten_leak.py --deals 60 --epochs 8
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
# meta 6 维在 108 维里的下标（102..107）：melds, riichi, discards, turn_norm, left, dora
META_SLICE = slice(N_TILES * 3, N_TILES * 3 + 6)
META_NAMES = ["melds", "riichi", "discards", "turn_norm", "left", "dora"]
FEAT_NAMES = (["obs_%d" % i for i in range(N_TILES)]
              + ["triv_%d" % i for i in range(N_TILES)]
              + ["oth_%d" % i for i in range(N_TILES)] + META_NAMES)


def order_planes(rivers):
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


def collect_raw(deals, seed=0):
    """采集原始分量（不拼特征），返回 dict of arrays。"""
    H, TR, OR_, MELD, RIICHI, DISC, LEFT, DORA, RIV, YSH, YTP = ([] for _ in range(11))
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
            o_cand = env.pending_players()
            if o_cand:
                o = o_cand[0]
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
                    triv = np.zeros(N_TILES, dtype=np.int8)
                    oth = np.zeros(N_TILES, dtype=np.int8)
                    for tile in rivers[t]:
                        triv[tile] += 1
                    for j in range(4):
                        if j in (o, t):
                            continue
                        for tile in rivers[j]:
                            oth[tile] += 1
                    H.append(obs_hand); TR.append(triv); OR_.append(oth)
                    MELD.append(len(tp.melds)); RIICHI.append(1 if tp.riichi else 0)
                    DISC.append(int(triv.sum()))
                    LEFT.append(len(env.wall) - env.wall_idx)
                    DORA.append(int(env.dora_indicator) if env.dora_indicator is not None else -1)
                    pad = [-1] * ORDER_LEN
                    RIV.append([ (rivers[p][-ORDER_LEN:] + pad)[:ORDER_LEN] for p in range(4)])
                    sh, tpl = shanten_label(counts)
                    YSH.append(sh); YTP.append(tpl)
            acts = {}
            for p in env.pending_players():
                legal = env.legal_actions(p)
                acts[p] = pols[p].choose(env, p, legal) if legal else E.PASS_ACTION
            r = env.step(acts)
        if (g + 1) % 25 == 0:
            print(f"  采集 {g+1}/{deals} 局 样本 {len(YSH)}", flush=True)
    return dict(
        hand=np.array(H, dtype=np.int8), triv=np.array(TR, dtype=np.int8),
        oth=np.array(OR_, dtype=np.int8), melds=np.array(MELD, dtype=np.float32),
        riichi=np.array(RIICHI, dtype=np.float32), discards=np.array(DISC, dtype=np.float32),
        left=np.array(LEFT, dtype=np.float32), dora=np.array(DORA, dtype=np.float32),
        riv=np.array(RIV, dtype=np.int8), ysh=np.array(YSH, dtype=np.int64),
        ytp=np.array(YTP, dtype=np.float32))


def build_matrix(raw, mode):
    """按 mode 拼 特征矩阵。mode in {base108, order, base_no_meta, base_no_time, meta_only}。"""
    n = len(raw["ysh"])
    meta = np.stack([raw["melds"] / 4.0, raw["riichi"], raw["discards"] / 18.0,
                     (136 - raw["left"]) / 136.0, raw["left"] / 136.0,
                     np.where(raw["dora"] >= 0, raw["dora"] / 33.0, -1.0)], axis=1).astype(np.float32)
    base = np.concatenate([raw["hand"].astype(np.float32) / 4.0,
                           raw["triv"].astype(np.float32) / 4.0,
                           raw["oth"].astype(np.float32) / 4.0, meta], axis=1)
    assert base.shape[1] == IN_DIM, base.shape
    if mode == "base108":
        return base
    if mode == "order":
        ordm = np.zeros((n, 4 * ORDER_LEN * N_TILES), dtype=np.float32)
        for i in range(n):
            ordm[i] = order_planes([list(r[r >= 0]) for r in raw["riv"][i]])
        return np.concatenate([base, ordm], axis=1)
    if mode == "base_no_meta":
        b = base.copy(); b[:, META_SLICE] = 0.0
        return b
    if mode == "base_no_time":
        b = base.copy()
        b[:, META_SLICE.start + 2:META_SLICE.start + 5] = 0.0  # discards, turn_norm, left
        return b
    if mode == "meta_only":
        return meta.copy()
    raise ValueError(mode)


class Head(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden // 2), nn.ReLU())
        self.sh = nn.Linear(hidden // 2, 7)
        self.tenpai = nn.Linear(hidden // 2, 1)

    def forward(self, x):
        h = self.net(x)
        return self.sh(h), self.tenpai(h).squeeze(-1)


def auc_score(scores, labels):
    labels = np.asarray(labels); scores = np.asarray(scores)
    order = np.argsort(scores)
    ranks = np.empty_like(order); ranks[order] = np.arange(len(scores))
    n_pos = labels.sum(); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))


def train_eval(X, Ysh, Ytp, turn, seed=0, epochs=8, bs=512, lr=1e-3):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = max(1, int(0.1 * len(X)))
    va, tr = idx[:n_val], idx[n_val:]
    model = Head(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.from_numpy(X[tr].astype(np.float32)); Sht = torch.from_numpy(Ysh[tr])
    Tpt = torch.from_numpy(Ytp[tr])
    Xv = torch.from_numpy(X[va].astype(np.float32)); Shv = Ysh[va]; Tpv = Ytp[va]
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
    pred_sh = sh.argmax(1).numpy(); scores = torch.sigmoid(tp).numpy()
    err = np.abs(pred_sh - Shv)
    out = {"mae": float(err.mean()), "auc": auc_score(scores, Tpv)}
    # 巡目分层 MAE（同巡目桶内看误差）
    tb = turn[va]
    strat = {}
    for lo, hi in [(0, 5), (6, 11), (12, 17), (18, 99)]:
        m = (tb >= lo) & (tb <= hi)
        if m.sum() > 50:
            strat[f"turn{lo}-{hi}"] = {"n": int(m.sum()), "mae": float(err[m].mean())}
    out["strat_mae"] = strat
    # 分层 macro-MAE（消除巡目分布差异后的误差）
    vals = [v["mae"] for v in strat.values()]
    out["macro_mae"] = float(np.mean(vals)) if vals else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=os.path.join(ROOT, "rules",
                     "OPP_SHANTEN_LEAK_2026-11-26.json"))
    args = ap.parse_args()
    torch.set_num_threads(2)

    print(f"[采集] {args.deals} 局 ...", flush=True)
    raw = collect_raw(args.deals, seed=args.seed)
    n = len(raw["ysh"])
    print(f"[采集完成] 样本 {n}  听牌占比 {raw['ytp'].mean():.4f}", flush=True)
    np.savez_compressed(os.path.join(ROOT, "data", f"oppsh_leak_{args.deals}.npz"),
                        **{k: v for k, v in raw.items()})
    turn = raw["discards"]

    res = {"deals": args.deals, "samples": n, "tenpai_rate": float(raw["ytp"].mean())}

    # ---- 1. 相关性 / 泄漏证据 ----
    base = build_matrix(raw, "base108")
    ysh_f = raw["ysh"].astype(np.float32); ytp = raw["ytp"]
    def col_r(X, y):
        Xc = X - X.mean(0); yc = y - y.mean()
        denom = np.sqrt((Xc ** 2).sum(0) * (yc ** 2).sum()) + 1e-12
        return (Xc * yc[:, None]).sum(0) / denom
    r_sh = col_r(base, ysh_f); r_tp = col_r(base, ytp)
    order_sh = np.argsort(-np.abs(r_sh))
    res["top15_feat_vs_shanten"] = [
        {"feat": FEAT_NAMES[i], "idx": int(i), "pearson_r_vs_shanten": round(float(r_sh[i]), 4),
         "pearson_r_vs_tenpai": round(float(r_tp[i]), 4)} for i in order_sh[:15]]
    res["meta_corr"] = {META_NAMES[k - META_SLICE.start]:
                        {"r_vs_shanten": round(float(r_sh[k]), 4),
                         "r_vs_tenpai": round(float(r_tp[k]), 4)}
                        for k in range(META_SLICE.start, META_SLICE.stop)}
    # 巡目 × 平均向听
    res["turn_vs_shanten"] = {}
    for lo, hi in [(0, 5), (6, 11), (12, 17), (18, 99)]:
        m = (turn >= lo) & (turn <= hi)
        if m.sum():
            res["turn_vs_shanten"][f"turn{lo}-{hi}"] = {
                "n": int(m.sum()), "mean_shanten": round(float(raw["ysh"][m].mean()), 3),
                "tenpai_rate": round(float(ytp[m].mean()), 4)}
    # 立直 × 听牌 列联表
    ri = raw["riichi"] > 0.5
    res["riichi_x_tenpai"] = {
        "riichi_n": int(ri.sum()), "riichi_tenpai_rate": round(float(ytp[ri].mean()), 4) if ri.sum() else None,
        "no_riichi_n": int((~ri).sum()), "no_riichi_tenpai_rate": round(float(ytp[~ri].mean()), 4),
        "tenpai_riichi_rate": round(float(ri[ytp > 0.5].mean()), 4) if (ytp > 0.5).sum() else None,
        "nontenpai_riichi_rate": round(float(ri[ytp < 0.5].mean()), 4)}

    # ---- 2/3. 各特征集训练评估 ----
    configs = [
        ("meta_only_6", "meta_only"),          # 捷径模型：只有巡目/立直/已打张等
        ("baseline_108", "base108"),           # 复现 baseline
        ("with_order_924", "order"),           # 复现 +顺序
        ("no_meta_102", "base_no_meta"),       # 控制泄漏：去掉全部 meta
        ("no_meta_plus_order_918", "order"),   # 去掉 meta + 顺序（顺序的真实增益）
        ("no_time_105", "base_no_time"),       # 只去 turn/left/discards，留 riichi
    ]
    res["models"] = {}
    mats = {}
    for name, mode in configs:
        if mode == "order" and "no_meta_plus_order_918" in name:
            # 去掉 meta 的 base + order
            b = build_matrix(raw, "base_no_meta")
            ordm = build_matrix(raw, "order")[:, IN_DIM:]
            X = np.concatenate([b, ordm], axis=1)
        elif mode == "order" and name == "no_time_plus_order":
            b = build_matrix(raw, "base_no_time")
            ordm = build_matrix(raw, "order")[:, IN_DIM:]
            X = np.concatenate([b, ordm], axis=1)
        else:
            X = build_matrix(raw, mode)
        mats[name] = X
        print(f"[训练] {name} dim={X.shape[1]} ...", flush=True)
        r = train_eval(X, raw["ysh"], raw["ytp"], turn, seed=args.seed, epochs=args.epochs)
        r["dim"] = int(X.shape[1])
        res["models"][name] = r
        print(f"   -> MAE={r['mae']:.4f} AUC={r['auc']:.4f} macroMAE={r['macro_mae']:.4f}", flush=True)

    # no_time + order 补跑
    b = build_matrix(raw, "base_no_time")
    ordm = build_matrix(raw, "order")[:, IN_DIM:]
    X = np.concatenate([b, ordm], axis=1)
    print(f"[训练] no_time_plus_order dim={X.shape[1]} ...", flush=True)
    r = train_eval(X, raw["ysh"], raw["ytp"], turn, seed=args.seed, epochs=args.epochs)
    r["dim"] = int(X.shape[1]); res["models"]["no_time_plus_order_921"] = r
    print(f"   -> MAE={r['mae']:.4f} AUC={r['auc']:.4f} macroMAE={r['macro_mae']:.4f}", flush=True)

    # ---- 增益汇总 ----
    def g(a, b_, key="mae"):
        return round(res["models"][a][key] - res["models"][b_][key], 4)
    res["gains"] = {
        "order_gain_full(108->924)_mae": g("baseline_108", "with_order_924"),
        "order_gain_full_auc": round(res["models"]["with_order_924"]["auc"] - res["models"]["baseline_108"]["auc"], 4),
        "order_gain_no_meta(102->918)_mae": g("no_meta_102", "no_meta_plus_order_918"),
        "order_gain_no_time(105->921)_mae": g("no_time_105", "no_time_plus_order_921"),
        "order_gain_no_meta_macro_mae": g("no_meta_102", "no_meta_plus_order_918", "macro_mae"),
        "order_gain_full_macro_mae": g("baseline_108", "with_order_924", "macro_mae"),
        "shortcut_meta_only_mae": res["models"]["meta_only_6"]["mae"],
        "shortcut_meta_only_auc": res["models"]["meta_only_6"]["auc"],
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    print("写入", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
