"""P3：独立「叫牌策略」（吃/碰/明杠/暗杠/加杠）——不塞进主策略。

数据：`dagger_collect --teacher call_first` 采的 47 槽数据里，凡是 mask 含叫牌槽
（PON39/MINKAN40/ADD_KAN41/ANKAN42/CHI44-46）的样本，标签=教师是否叫、叫哪种。
训练一个小 MLP（obs 786+touch8 → 叫牌概率 + 类型），推理时与主策略组合：
弃牌仍走主模型，遇到可叫牌时用本策略决定「叫/过」。

用法：
    python call_strategy.py --train data/dagger_call300.npz
    python call_strategy.py --eval checkpoints/call_policy.pt --deals 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cpu_only  # noqa
from arena_elo import RuleAgent, CKPT
from mahjong import env as E
from mahjong.hanchan import HanchanState
from version_benchmark import make_agent

CALL_SLOTS = {39: "pon", 40: "minkan", 41: "add_kan", 42: "ankan",
              44: "chi", 45: "chi", 46: "chi"}
TYPES = ["pon", "minkan", "add_kan", "ankan", "chi"]


class CallNet(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden // 2), nn.ReLU())
        self.call = nn.Linear(hidden // 2, 1)
        self.type_head = nn.Linear(hidden // 2, len(TYPES))

    def forward(self, x):
        h = self.trunk(x)
        return self.call(h).squeeze(-1), self.type_head(h)


def train(data_path, out):
    d = np.load(data_path)
    X = np.concatenate([d["x"], d["tf"]], axis=1).astype(np.float32)
    mask = d["mask"]; y = d["y"]
    call_slots = set(CALL_SLOTS)
    sel = []
    for i in range(len(X)):
        slots = set(np.nonzero(mask[i])[0].tolist())
        if slots & call_slots:
            sel.append(i)
    sel = np.array(sel)
    Xc = X[sel]; yc = y[sel]
    labels = np.array([1.0 if int(t) in call_slots else 0.0 for t in yc],
                      dtype=np.float32)
    types = np.array([TYPES.index(CALL_SLOTS[int(t)]) if int(t) in call_slots
                      else 0 for t in yc], dtype=np.int64)
    print(f"叫牌合法样本 {len(Xc)}，其中教师叫牌 {int(labels.sum())}"
          f"（{100*labels.mean():.1f}%）", flush=True)
    torch.set_num_threads(2)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(Xc)); nv = max(1, int(0.15 * len(Xc)))
    va, tr = idx[:nv], idx[nv:]
    net = CallNet(Xc.shape[1])
    opt = torch.optim.Adam(net.parameters(), 3e-4)
    Xt = torch.from_numpy(Xc[tr]); Lt = torch.from_numpy(labels[tr]); Tt = torch.from_numpy(types[tr])
    Xv = torch.from_numpy(Xc[va]); Lv = torch.from_numpy(labels[va]); Tv = torch.from_numpy(types[va])
    pos_w = float((1 - labels.mean()) / max(1e-6, labels.mean()))
    for ep in range(15):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            b = perm[i:i + 256]
            pc, pt = net(Xt[b])
            loss = nn.functional.binary_cross_entropy_with_logits(
                pc, Lt[b], pos_weight=torch.tensor(pos_w)) + 0.5 * \
                nn.functional.cross_entropy(pt, Tt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pc, pt = net(Xv)
        acc = float(((torch.sigmoid(pc) > 0.5).float() == Lv).float().mean())
        call_rec = float(((torch.sigmoid(pc) > 0.5).float() * Lv).sum() /
                         max(1.0, Lv.sum()))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "in_dim": int(Xc.shape[1])}, out)
    print(f"val 叫牌 acc={acc:.3f} recall={call_rec:.3f} → {out}")
    return 0


def _is_call(a):
    return a.type in (E.ACT_PON, E.ACT_MINKAN, E.ACT_ANKAN, E.ACT_ADD_KAN, E.ACT_CHI)


class CallAgent:
    """主模型管弃牌/和了，叫牌用独立 CallNet。"""

    kind = "call_agent"

    def __init__(self, name, main_ckpt, call_ckpt, threshold=0.35):
        self.main = make_agent(name, main_ckpt)
        self.hstate = None
        ck = torch.load(call_ckpt, map_location="cpu", weights_only=False)
        self.net = CallNet(ck["in_dim"])
        self.net.load_state_dict(ck["state_dict"]); self.net.eval()
        self.threshold = threshold
        self.enc = None

    def set_game(self, env, hstate):
        self.hstate = hstate
        if hasattr(self.main, "set_game"):
            self.main.set_game(env, hstate)

    def act(self, env, p_idx, legal):
        from train_hanchan import encode
        from mahjong import obs2 as O2
        calls = [a for a in legal if _is_call(a)]
        if calls:
            if self.enc is None:
                self.enc = O2.ObsEncoder()
            vec, sh, sit, gu, tf = encode(env, self.enc, self.hstate, p_idx)
            x = np.concatenate([vec, tf]).astype(np.float32)
            with torch.no_grad():
                pc, pt = self.net(torch.from_numpy(x)[None])
            prob = float(torch.sigmoid(pc)[0])
            if prob >= self.threshold:
                pref = TYPES[int(pt[0].argmax())]
                for a in calls:
                    name = ("pon" if a.type == E.ACT_PON else
                            "minkan" if a.type == E.ACT_MINKAN else
                            "ankan" if a.type == E.ACT_ANKAN else
                            "add_kan" if a.type == E.ACT_ADD_KAN else "chi")
                    if name == pref:
                        return a
            return E.PASS_ACTION
        return self.main.act(env, p_idx, legal)


def evaluate(main_ckpt, call_ckpt, deals=20, seed=77):
    from train_hanchan import encode  # noqa
    with_calls = [CallAgent("teacher47_100h_v2", main_ckpt, call_ckpt)
                  for _ in range(4)]
    no_calls = [make_agent("teacher47_100h_v2", main_ckpt) for _ in range(4)]
    out = {}
    for label, agents in (("call_agent", with_calls), ("no_call", no_calls)):
        calls = 0; wins = 0; ron = 0; tsumo = 0
        for g in range(deals):
            h = HanchanState(dealer=g % 4); env = h.new_env(seed=seed + g * 131)
            for a in agents:
                a.set_game(env, h)
            r = env.start_turn()
            while r is None:
                acts = {}
                for p in env.pending_players():
                    legal = env.legal_actions(p)
                    if not legal:
                        acts[p] = E.PASS_ACTION
                        continue
                    act = agents[p].act(env, p, legal)
                    if _is_call(act):
                        calls += 1
                    acts[p] = act
                r = env.step(acts)
            if r.result_type == "ron":
                ron += 1
            elif r.result_type == "tsumo":
                tsumo += 1
        out[label] = {"deals": deals, "calls": calls, "ron": ron,
                      "tsumo": tsumo, "deal_win_rate": round(100*(ron+tsumo)/deals, 2)}
        print(label, out[label], flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=None)
    ap.add_argument("--eval", default=None)
    ap.add_argument("--main-ckpt", default=os.path.join(CKPT, "teacher47_100h_v2.pt"))
    ap.add_argument("--out", default=os.path.join(CKPT, "call_policy.pt"))
    args = ap.parse_args()
    if args.train:
        return train(args.train, args.out)
    if args.eval:
        out = evaluate(args.main_ckpt, args.eval)
        with open("rules/CALL_STRATEGY_2026-11-05.json", "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        return 0
    print("用法 --train 或 --eval")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
