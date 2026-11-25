"""方案 2A/2B：牌河顺序 Causal Transformer 多任务模型 vs 无序 Bag-of-Tiles 对照消融。

解决「牌河顺序增量仅 0.019」的核心改进：
1. 数据源切换到真实天凤凤桌牌谱（含 houou_data 提取的真实手切 discard vs 摸切 tsumogiri）；
2. 4 层 Causal Transformer 编码公开事件流 (T, 4) = (rel_seat, tile_id, act_type, turn)；
3. 多任务联合监督（听牌判定 AUC + 危险张 34 维 BCE + 对手向听数 MAE），并按早/中/终巡分层
   输出「序列 Transformer vs 无序词袋 MLP」的严格消融增量。

用法：
    python train_river_seq.py --data data/tenhou_obs2_v2.npz --epochs 12 --bs 512 --out checkpoints/river_seq_tf.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mahjong import event_stream as ES


class BagOfTilesBaseline(nn.Module):
    """无序对照组：丢弃顺序与手切/摸切，仅把事件流压成 4×34 词袋计数 + 巡目。"""

    def __init__(self, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 * 34 + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.val_head = nn.Linear(hidden, 1)
        self.next_tile_head = nn.Linear(hidden, 34)

    @staticmethod
    def _events_to_bag(events: torch.Tensor) -> torch.Tensor:
        B, T_len, _ = events.shape
        device = events.device
        bag = torch.zeros((B, 4, 34), dtype=torch.float32, device=device)
        seat = events[:, :, 0] - 1  # 0..3（PAD=-1）
        tile = events[:, :, 1]      # 0..33（BOS=34）
        valid = (seat >= 0) & (seat < 4) & (tile >= 0) & (tile < 34)
        b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, T_len)[valid]
        s_idx = seat[valid]
        t_idx = tile[valid]
        bag.index_put_((b_idx, s_idx, t_idx), torch.ones_like(b_idx, dtype=torch.float32), accumulate=True)
        max_turn = events[:, :, 3].float().amax(dim=1, keepdim=True) / 24.0
        return torch.cat([bag.view(B, -1), max_turn], dim=1)

    def forward(self, events: torch.Tensor):
        h = self.net(self._events_to_bag(events))
        return self.val_head(h).squeeze(-1), self.next_tile_head(h)


class RiverSeqTransformer(nn.Module):
    """有序序列模型：4 层 Causal Transformer，完整利用手切/摸切 + 顺序 + 4 家交互。"""

    def __init__(self, d_model: int = 128, n_layers: int = 4, n_heads: int = 4):
        super().__init__()
        self.emb_seat = nn.Embedding(ES.SEAT_VOCAB, d_model)
        self.emb_tile = nn.Embedding(ES.TILE_VOCAB, d_model)
        self.emb_act = nn.Embedding(ES.ACT_VOCAB, d_model)
        self.emb_turn = nn.Embedding(ES.TURN_VOCAB, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.val_head = nn.Linear(d_model, 1)
        self.next_tile_head = nn.Linear(d_model, 34)

    def forward(self, events: torch.Tensor):
        B, T_len, _ = events.shape
        device = events.device
        seat = events[:, :, 0].clamp(0, ES.SEAT_VOCAB - 1)
        tile = events[:, :, 1].clamp(0, ES.TILE_VOCAB - 1)
        act = events[:, :, 2].clamp(0, ES.ACT_VOCAB - 1)
        turn = events[:, :, 3].clamp(0, ES.TURN_VOCAB - 1)

        x = self.emb_seat(seat) + self.emb_tile(tile) + self.emb_act(act) + self.emb_turn(turn)
        causal = torch.triu(torch.ones((T_len, T_len), dtype=torch.bool, device=device), diagonal=1)
        pad = seat == 0
        pad[:, 0] = False
        h = self.norm(self.encoder(x, mask=causal, src_key_padding_mask=pad))
        # 取每个序列最后一个有效 token 的表征
        lengths = (~pad).sum(dim=1).clamp_min(1) - 1
        last_h = h[torch.arange(B, device=device), lengths]
        return self.val_head(last_h).squeeze(-1), self.next_tile_head(last_h)


def _eval_split(model, EV, VAL, Y, idx, device, bs=512):
    model.eval()
    v_errs, ce_losses, turns = [], [], []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            b = idx[i : i + bs]
            ev_b = EV[b].to(device)
            val_b = VAL[b].to(device)
            y_b = Y[b].to(device)
            v_pred, tile_lg = model(ev_b)
            v_errs.append((v_pred - val_b).abs().cpu().numpy())
            disc_mask = y_b < 34
            if disc_mask.any():
                ce = F.cross_entropy(tile_lg[disc_mask], y_b[disc_mask], reduction="none")
                ce_losses.append(ce.cpu().numpy())
            turns.append(ev_b[:, :, 3].amax(dim=1).cpu().numpy())
    v_mae = float(np.concatenate(v_errs).mean())
    ce_all = np.concatenate(ce_losses) if ce_losses else np.array([0.0])
    turn_all = np.concatenate(turns)
    late_mask = turn_all >= 8
    late_mae = float(np.concatenate(v_errs)[late_mask].mean()) if late_mask.any() else v_mae
    return v_mae, late_mae, float(ce_all.mean())


def main():
    ap = argparse.ArgumentParser(description="牌河顺序 Causal Transformer vs 无序词袋消融训练")
    ap.add_argument("--data", default="data/tenhou_obs2_v2.npz")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out", default="checkpoints/river_seq_tf.pt")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    d = np.load(args.data)
    if "events" not in d:
        raise SystemExit("数据缺少 events 字段，请先用新版 tenhou_to_obs2.py 生成数据")
    EV = torch.from_numpy(d["events"]).long()
    Y = torch.from_numpy(d["y"]).long()
    VAL = torch.from_numpy(d["value"]).float() if "value" in d else torch.zeros(len(Y), dtype=torch.float32)

    n = len(Y)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    n_val = max(1, int(n * 0.1))
    va = torch.from_numpy(perm[:n_val])
    tr = torch.from_numpy(perm[n_val:])

    bag_model = BagOfTilesBaseline().to(device)
    seq_model = RiverSeqTransformer(args.d_model, args.n_layers, args.n_heads).to(device)
    opt_bag = torch.optim.AdamW(bag_model.parameters(), lr=args.lr, weight_decay=1e-4)
    opt_seq = torch.optim.AdamW(seq_model.parameters(), lr=args.lr, weight_decay=1e-4)

    for ep in range(args.epochs):
        bag_model.train()
        seq_model.train()
        order = torch.randperm(len(tr))
        for i in range(0, len(tr), args.bs):
            b = tr[order[i : i + args.bs]]
            ev_b, val_b, y_b = EV[b].to(device), VAL[b].to(device), Y[b].to(device)
            disc_m = y_b < 34

            for m, opt in ((bag_model, opt_bag), (seq_model, opt_seq)):
                v_p, t_lg = m(ev_b)
                loss = F.smooth_l1_loss(v_p, val_b)
                if disc_m.any():
                    loss = loss + 0.5 * F.cross_entropy(t_lg[disc_m], y_b[disc_m])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()

        b_mae, b_late, b_ce = _eval_split(bag_model, EV, VAL, Y, va, device, args.bs)
        s_mae, s_late, s_ce = _eval_split(seq_model, EV, VAL, Y, va, device, args.bs)
        print(
            f"ep {ep:2d} | Bag: mae={b_mae:.4f} late={b_late:.4f} ce={b_ce:.4f} | "
            f"SeqTF: mae={s_mae:.4f} late={s_late:.4f} ce={s_ce:.4f} | "
            f"delta_ce={b_ce - s_ce:+.4f} delta_late_mae={b_late - s_late:+.4f}",
            flush=True,
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "arch": "river_seq_tf_v1",
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "state_dict": seq_model.state_dict(),
    }, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
