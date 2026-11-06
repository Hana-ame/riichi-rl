"""副露（吃/碰/杠）独立二分类头：用模型决定「叫/不叫」，不靠规则。

动机（2026-11-06）：47 槽主策略被 pass 类标签淹没（120 局只叫 1 次），
叫牌是「低频 + 非支配」动作，BC 学不出。改为**单独的二分类器**：
输入 obs(786)+touch(8)，输出 P(叫)，训练集从「叫牌决策点」样本里平衡采样。
叫牌具体种类/用哪张：由主模型在合法叫牌槽上的 logits 取最大（仍是模型输出）。

训练：
  python call_head.py train --data data/hybrid_lmcall60.npz --out checkpoints/call_head.pt
评测（4 席同池，弃牌/立直/和牌=eff 主模型，叫牌=二分类头）：
  python eval_selfplay_gate.py --ckpt teacher47_eff_v3 --call-head checkpoints/call_head.pt --call-thr 0.5 --deals 120
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CALL_SLOTS = [39, 40, 41, 42, 44, 45, 46]   # pon/minkan/add_kan/ankan/chi x3


class CallHead(nn.Module):
    def __init__(self, in_dim=794, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)


def _vec(x, tf):
    return np.concatenate([x.reshape(len(x), -1), tf.reshape(len(tf), -1)], axis=1).astype(np.float32)


def train_main(a):
    d = np.load(a.data)
    X, TF, M, Y = d["x"], d["tf"], d["mask"], d["y"]
    has = M[:, CALL_SLOTS].any(axis=1)
    lab = np.isin(Y, CALL_SLOTS).astype(np.int64)
    X, TF, lab = X[has], TF[has], lab[has]
    V = _vec(X, TF)
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(V))
    V, lab = V[idx], lab[idx]
    n_val = max(1, len(V) // 10)
    Xtr, ytr, Xva, yva = V[n_val:], lab[n_val:], V[:n_val], lab[:n_val]
    w = torch.tensor([len(ytr) / max(1, (ytr == 0).sum()),
                      len(ytr) / max(1, (ytr == 1).sum())], dtype=torch.float32)
    model = CallHead(in_dim=V.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss(weight=w)
    Xt = torch.from_numpy(Xtr); yt = torch.from_numpy(ytr)
    Xv = torch.from_numpy(Xva); yv = torch.from_numpy(yva)
    bs = 256
    for ep in range(12):
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(Xt), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xt[b]), yt[b])
            loss.backward(); opt.step()
            tot += float(loss) * len(b)
        with torch.no_grad():
            pr = model(Xv).argmax(1)
            acc = float((pr == yv).float().mean())
            p1 = float((pr == 1).float().mean())
        print(f"epoch {ep} loss={tot/len(Xt):.4f} val_acc={acc:.3f} pred_pos={p1:.3f}", flush=True)
    torch.save({"state_dict": model.state_dict(), "in_dim": int(V.shape[1]),
                "hidden": 256, "thr": a.thr}, a.out)
    print(f"保存 {a.out}；正样本率 {float(lab.mean()):.3f}，机会样本 {len(V)}")


class CallHeadAgent:
    """主模型（47 槽）+ 叫牌二分类头。"""
    def __init__(self, name, ckpt, head_ckpt, thr=0.5):
        from version_benchmark import AutoAgent
        from train_hanchan import slot_of
        self.base = AutoAgent(name, ckpt)
        self.TH = self.base.TH
        self.enc = self.base.enc
        self.hstate = None
        self.thr = float(thr)
        ck = torch.load(head_ckpt, map_location="cpu", weights_only=False)
        self.head = CallHead(in_dim=int(ck["in_dim"]), hidden=int(ck.get("hidden", 256)))
        self.head.load_state_dict(ck["state_dict"])
        self.head.eval()
        self.calls = 0

    def set_game(self, env, hstate=None):
        self.hstate = hstate
        self.base.set_game(env, hstate)

    def _p_call(self, env, p):
        from train_hanchan import encode
        vec, sh, sit, gu, tf = encode(env, self.enc, self.hstate, p)
        V = _vec(vec[None], tf.reshape(1, -1))
        with torch.no_grad():
            logit = self.head(torch.from_numpy(V))[0]
            return float(torch.softmax(logit, -1)[1])

    def act(self, env, p, legal):
        from mahjong import env as E
        from version_benchmark import _rule_override
        ov = _rule_override(legal)
        if ov is not None:
            return ov
        calls = [a for a in legal if a.type in (E.ACT_CHI, E.ACT_PON,
                                                E.ACT_MINKAN, E.ACT_ANKAN,
                                                E.ACT_ADD_KAN)]
        if calls and env.phase == E.Phase.RESPONSE:
            if self._p_call(env, p) >= self.thr:
                # 具体叫牌种类由主模型 logits 选（仍是模型输出）
                from train_hanchan import slot_of
                logits, slots = self._logits(env, p, legal)
                best, bs = None, -1e18
                for a in calls:
                    s = slot_of(a) if self.base.is47 else slots[legal.index(a)]
                    if 0 <= s < len(logits) and float(logits[s]) > bs:
                        bs, best = float(logits[s]), a
                if best is not None:
                    self.calls += 1
                    return best
            return E.PASS_ACTION if E.PASS_ACTION in legal else legal[0]
        return self.base.act(env, p, legal)

    def _logits(self, env, p, legal):
        from version_benchmark import slot_of54
        from train_hanchan import slot_of
        slot_fn = slot_of if self.base.is47 else slot_of54
        vec, sh, sit, gu, tf = self.TH.encode(env, self.enc, self.hstate, p)
        legal2 = [a for a in legal if slot_fn(a) >= 0]
        with torch.no_grad():
            lg = self.base.model(torch.from_numpy(vec).unsqueeze(0),
                                 torch.from_numpy(tf).unsqueeze(0))[0]
        return lg.reshape(-1).numpy(), [slot_fn(a) for a in legal2]


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default="data/hybrid_lmcall60.npz")
    t.add_argument("--out", default="checkpoints/call_head.pt")
    t.add_argument("--thr", type=float, default=0.5)
    t.set_defaults(func=train_main)
    a = ap.parse_args()
    torch.set_num_threads(4)
    a.func(a)


if __name__ == "__main__":
    main()
