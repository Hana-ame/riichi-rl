"""泄漏诊断补充分析（2026-11-26）：order-only 消融 + 增益显著性（分层 bootstrap）。

需要 data/oppsh_leak_{deals}.npz（opp_shanten_leak.py 产出）。
  - order_only_816：只喂牌河顺序（无任何手牌/牌河 counts/meta）→ 看顺序自身信号
  - order+meta_only：顺序 + meta6（无 counts）→ 泄漏捷径 + 顺序
  - 增益显著性：在验证集上对 baseline vs +顺序 的绝对误差差做分层 bootstrap 95%CI
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn
from opponent_model import IN_DIM, N_TILES
from opp_shanten_leak import (Head, ORDER_LEN, META_SLICE, META_NAMES,
                              build_matrix, train_eval, auc_score)

def load(deals):
    d = np.load(os.path.join(ROOT, "data", f"oppsh_leak_{deals}.npz"))
    return {k: d[k] for k in d.files}

ROOT = os.path.dirname(os.path.abspath(__file__))

def main():
    deals = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    seed = 2026
    torch.set_num_threads(2)
    raw = load(deals)
    n = len(raw["ysh"]); turn = raw["discards"]
    base = build_matrix(raw, "base108")
    ordm = build_matrix(raw, "order")[:, IN_DIM:]
    meta = build_matrix(raw, "meta_only")

    res = {"deals": deals, "samples": n}
    # order_only（只有顺序 816 维）
    X = ordm.copy()
    print(f"[训练] order_only dim={X.shape[1]}", flush=True)
    r = train_eval(X, raw["ysh"], raw["ytp"], turn, seed=seed, epochs=epochs)
    r["dim"] = X.shape[1]; res["order_only_816"] = r
    print(f"  -> MAE={r['mae']:.4f} AUC={r['auc']:.4f} macro={r['macro_mae']:.4f}", flush=True)
    # order + meta（无 counts）
    X2 = np.concatenate([meta, ordm], axis=1)
    print(f"[训练] order+meta dim={X2.shape[1]}", flush=True)
    r2 = train_eval(X2, raw["ysh"], raw["ytp"], turn, seed=seed, epochs=epochs)
    r2["dim"] = X2.shape[1]; res["order_plus_meta_822"] = r2
    print(f"  -> MAE={r2['mae']:.4f} AUC={r2['auc']:.4f} macro={r2['macro_mae']:.4f}", flush=True)

    # ---- 增益显著性：分层 bootstrap（按巡目桶，验证集固定切分）----
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n); n_val = max(1, int(0.1*n))
    va = idx[:n_val]
    def err_of(X):
        m = train_eval(X, raw["ysh"], raw["ytp"], turn, seed=seed, epochs=epochs)
        return m  # only to get model; redo manually below
    # 手动：训练两个模型并取同一 va 的逐样本误差
    def abs_err(X):
        from opp_shanten_leak import Head
        tr = idx[n_val:]
        model = Head(X.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        Xt = torch.from_numpy(X[tr].astype(np.float32))
        Sht = torch.from_numpy(raw["ysh"][tr]); Tpt = torch.from_numpy(raw["ytp"][tr])
        for ep in range(epochs):
            model.train(); perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 512):
                b = perm[i:i+512]; sh, tp = model(Xt[b])
                loss = nn.functional.cross_entropy(sh, Sht[b]) + \
                    nn.functional.binary_cross_entropy_with_logits(tp, Tpt[b])
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            sh, tp = model(torch.from_numpy(X[va].astype(np.float32)))
        return np.abs(sh.argmax(1).numpy() - raw["ysh"][va]).astype(float)
    e_base = abs_err(base); e_ord = abs_err(np.concatenate([base, ordm], axis=1))
    no_meta = np.concatenate([base[:, :META_SLICE.start], base[:, META_SLICE.stop:]], axis=1)
    e_nm = abs_err(no_meta)
    e_nm_ord = abs_err(np.concatenate([np.concatenate([base[:, :META_SLICE.start], base[:, META_SLICE.stop:]], axis=1), ordm], axis=1))
    tb = turn[va]
    buckets = [(0,5),(6,11),(12,17),(18,99)]
    boot = {"order_gain_full": [], "order_gain_no_meta": []}
    for _ in range(2000):
        parts_f, parts_n = [], []
        for lo,hi in buckets:
            m = (tb>=lo)&(tb<=hi)
            if m.sum()==0: continue
            sel = rng.choice(np.where(m)[0], size=int(m.sum()), replace=True)
            parts_f.append(e_ord[sel]-e_base[sel])
            parts_n.append(e_nm_ord[sel]-e_nm[sel])
        boot["order_gain_full"].append(np.concatenate(parts_f).mean())
        boot["order_gain_no_meta"].append(np.concatenate(parts_n).mean())
    def ci(v):
        v=np.array(v); return {"mean":round(float(v.mean()),4),
            "lo95":round(float(np.percentile(v,2.5)),4),"hi95":round(float(np.percentile(v,97.5)),4)}
    res["gain_ci"] = {
        "order_gain_full_mae(baseline->+order)": ci(boot["order_gain_full"]),
        "order_gain_no_meta_mae(no_meta->+order)": ci(boot["order_gain_no_meta"]),
        "order_gain_full_strat_macro_mae": round(float((e_ord-e_base).mean()),4),
    }
    res["val_subset_mae"] = {"baseline_108":round(float(e_base.mean()),4),
                             "with_order_924":round(float(e_ord.mean()),4),
                             "no_meta_102":round(float(e_nm.mean()),4),
                             "no_meta_plus_order_918":round(float(e_nm_ord.mean()),4)}
    out = os.path.join(ROOT,"rules","OPP_SHANTEN_LEAK_EXTRA.json")
    json.dump(res, open(out,"w",encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k:res[k] for k in ("order_only_816","order_plus_meta_822","gain_ci","val_subset_mae")}, ensure_ascii=False, indent=1))
    print("写入", out)

if __name__ == "__main__":
    main()
