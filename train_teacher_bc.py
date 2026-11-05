"""在「固定槽位」教师数据上做带特殊动作加权的行为克隆（47 槽半庄线）。

配套：`gen_teacher_data.py` 用 GreedyTenpai 教师生成 (obs, touch, 合法 mask, 槽标签)。
本脚本训练 `MahjongNet(channels=64, blocks=4, n_actions=47, touch_dim=8,
extra_scalar=12)`，special 槽（TSUMO=34/RON=35/RIICHI=36）样本损失加权，
避免「和牌/立直动作占比 ~1–3% 被 argmax 忽略」——这是纯模型和牌率低的主因之一。

产出 ckpt 与 `eval_hanchan.load_model` 兼容（含 channels/blocks 字段）。

用法：
    python train_teacher_bc.py --data data/teacher47.npz \
        --epochs 6 --special-weight 20 --out checkpoints/teacher47_v1.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mahjong.net import MahjongNet, count_params
from mahjong.touch import TOUCH_DIM
from train_hanchan import N_ACTIONS, SIT_DIM

SPECIAL_SLOTS = (34, 35, 36)      # TSUMO / RON / RIICHI（固定槽）


def accuracy(logits, y):
    return float((logits.argmax(-1) == y).float().mean())


def class_accuracy(logits, y, mask_y):
    if int(mask_y.sum()) == 0:
        return float("nan")
    pred = logits.argmax(-1)
    return float((pred[mask_y] == y[mask_y]).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/teacher47.npz")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--special-weight", type=float, default=20.0)
    ap.add_argument("--discard-weight", type=float, default=3.0,
                    help="打牌样本权重：PASS 占 75%% 数据，不调权模型只会学 PASS")
    ap.add_argument("--pass-weight", type=float, default=0.1,
                    help="响应 PASS 样本权重（最简单、占比最高，压低）")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--init", default="", help="从已有 ckpt 继续训练")
    ap.add_argument("--dora-aux-weight", type=float, default=0.0,
                    help="dora aux 任务 BCE 权重（0=关闭）")
    ap.add_argument("--out", default="checkpoints/teacher47_v1.pt")
    args = ap.parse_args()

    d = np.load(args.data)
    X = torch.from_numpy(d["x"]).float()
    TF = torch.from_numpy(d["tf"]).float()
    M = torch.from_numpy(d["mask"])
    Y = torch.from_numpy(d["y"]).long()
    n = len(X)
    print(f"样本 {n}  obs={X.shape[1]}  touch={TF.shape[1]}  "
          f"权重 discard={args.discard_weight} pass={args.pass_weight} "
          f"special={args.special_weight}", flush=True)

    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    n_val = max(1, int(n * 0.1))
    tr, va = perm[n_val:], perm[:n_val]
    tr = torch.from_numpy(tr); va = torch.from_numpy(va)

    _xdim = int(X.shape[1])
    # x = 平面*34 + 标量26 + 局势12；touch 单独在 tf（不属于 x）
    _in_planes = (_xdim - 26 - SIT_DIM) // 34
    aux_labels = d["aux"].astype(np.float32) if "aux" in d else None
    if aux_labels is not None and len(aux_labels) != len(Y):
        print(f"[warn] aux 长度 {len(aux_labels)} != y 长度 {len(y)}，补零/截断对齐", flush=True)
        if len(aux_labels) < len(Y):
            aux_labels = np.pad(aux_labels, (0, len(Y) - len(aux_labels)))
        else:
            aux_labels = aux_labels[:len(Y)]
    print(f"[debug] aux_labels len={None if aux_labels is None else len(aux_labels)}", flush=True)
    aux_dim = 1 if (args.dora_aux_weight > 0 and aux_labels is not None) else 0
    model = MahjongNet(channels=args.channels, n_blocks=args.blocks,
                       n_actions=N_ACTIONS, touch_dim=TOUCH_DIM,
                       extra_scalar=SIT_DIM, in_planes=_in_planes, aux_dim=aux_dim)
    obs_order = _in_planes > 22
    print(f"obs planes={_in_planes} obs_order={obs_order}", flush=True)
    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"], strict=False)
        print(f"从 {args.init} 继续训练（val_acc={ck.get('best_val_acc')}）",
              flush=True)
    print(f"params={count_params(model):,} device=cpu", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    yva = Y[va]

    def sample_weight(y):
        w = torch.ones(len(y))
        w[y < 34] = args.discard_weight
        w[y == 43] = args.pass_weight          # PASS
        for sl in SPECIAL_SLOTS:
            w[y == sl] = args.special_weight
        return w

    best = -1.0
    for ep in range(args.epochs):
        model.train()
        p = torch.randperm(len(tr))
        tot_loss = 0.0
        for i in range(0, len(tr), args.bs):
            b = tr[p[i:i + args.bs]]
            if aux_dim:
                logits, _, auxl = model.forward_aux(X[b], TF[b])
            else:
                logits, _ = model(X[b], TF[b])
            logits = logits.masked_fill(~M[b], -1e9)
            ce = F.cross_entropy(logits, Y[b], reduction="none")
            w = sample_weight(Y[b])
            loss = (ce * w).sum() / w.sum()
            if aux_dim:
                # b 已经是全局样本索引（来自 tr），直接索引即可
                auxb = torch.from_numpy(aux_labels[b.numpy()])
                loss = loss + args.dora_aux_weight * F.binary_cross_entropy_with_logits(
                    auxl.squeeze(-1), auxb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += float(loss) * len(b)

        model.eval()
        with torch.no_grad():
            lv, _ = model(X[va], TF[va])
            lv = lv.masked_fill(~M[va], -1e9)
            v_loss = float(F.cross_entropy(lv, yva, reduction="none").mean())
            v_acc = accuracy(lv, yva)
            v_special_rec = class_accuracy(
                lv, yva, torch.tensor([int(v) in SPECIAL_SLOTS for v in yva]))
        print(f"epoch {ep:>2}  train_loss={tot_loss / len(tr):.4f}  "
              f"val_loss={v_loss:.4f}  val_acc={v_acc:.3f}  "
              f"special_recall={v_special_rec:.3f}", flush=True)
        if v_acc > best:
            best = v_acc
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "channels": args.channels,
                "blocks": args.blocks,
                "n_actions": N_ACTIONS,
                "in_planes": int(_in_planes),
                "obs_order": bool(obs_order),
                "aux_dim": int(aux_dim),
                "touch_dim": TOUCH_DIM,
                "extra_scalar": SIT_DIM,
                "iter": ep + 1,
                "deals": -1,
                "win_rate": 0.0,
                "samples": n,
                "best_val_acc": best,
            }, args.out)
    print(f"saved {args.out} (best val_acc={best:.3f})")


if __name__ == "__main__":
    main()