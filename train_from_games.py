"""用观战对弈记录（data/selfplay/*.npz）训练 MahjongNet。

数据是 serve_game.py 观战服务器逐决策点记录的：obs(774维) + act(槽号) + seat + pol(谁出的)。
两种用法：
  --pol llm    只用 LLM 的决策 → 蒸馏 LLM（LLM 当老师，上限就是它的水平）
  --pol all    全部决策混用（含规则机器人，质量参差，不推荐单独用）

纯 RL 自我对弈训练（不模仿任何人）仍是 train_ppo.py，两者可并行对比。

用法：
    /mnt/d/WorkPlace/my_rl_project/venv/bin/python train_from_games.py --pol llm --epochs 5
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mahjong.net import MahjongNet, count_params
from train_reinforce import N_ACTIONS

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_dataset(pol_filter, data_dir=None):
    X, Y = [], []
    d = data_dir or os.path.join(ROOT, "data", "selfplay")
    files = sorted(glob.glob(os.path.join(d, "*.npz")))
    for f in files:
        d = np.load(f, allow_pickle=True)
        if pol_filter and pol_filter != "all":
            m = d["pol"] == pol_filter
        else:
            m = np.ones(len(d["act"]), dtype=bool)
        if m.any():
            X.append(d["obs"][m])
            Y.append(d["act"][m])
    if not X:
        return None, None, 0
    return np.concatenate(X), np.concatenate(Y), len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pol", default="llm", help="llm / net / greedy_tenpai / all")
    ap.add_argument("--data", default="", help="数据目录，默认 data/selfplay")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="checkpoints/from_games_v1.pt")
    args = ap.parse_args()

    X, Y, n_files = load_dataset(args.pol, args.data or None)
    if X is None:
        print(f"没有数据：data/selfplay/ 里没有 pol={args.pol} 的记录。"
              f"先开着观战界面跑几局再训。")
        return 1
    print(f"数据：{len(X)} 决策点 来自 {n_files} 局  pol={args.pol}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = MahjongNet().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    Xt = torch.from_numpy(X).float()
    Yt = torch.from_numpy(Y).long()
    print(f"device={dev} params={count_params(model):,} "
          f"动作空间={N_ACTIONS}", flush=True)

    n = len(Xt)
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        tot = cor = 0
        loss_sum = 0.0
        for i in range(0, n, args.bs):
            idx = perm[i:i + args.bs]
            xb, yb = Xt[idx].to(dev), Yt[idx].to(dev)
            logits, _ = model(xb)
            loss = torch.nn.functional.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += float(loss) * len(idx)
            cor += int((logits.argmax(-1) == yb).sum())
            tot += len(idx)
        print(f"epoch {ep}  loss={loss_sum / tot:.4f}  "
              f"acc={cor / tot:.3f}（与记录动作一致率）", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "arch": "convformer_v1",
                "trained_on": f"selfplay pol={args.pol} {n_files}局 {n}决策点"},
               args.out)
    print(f"saved {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
