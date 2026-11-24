"""
Tenhou 离线 BC：从 houou_data.Decision 直接提特征，训练 MLP 动作分类器。

数据：data/houou/*.tar（1nfrared218/tenhou-mjai-riichi-training，CC-BY-4.0）
特征（560 维）：
  0-33   自己手牌计数（34）
  34-67  自己弃牌计数（34）
  68-203 四家弃牌计数（4×34=136）
  204-339 四家副露计数（4×34，chi算3张）
  340-509 dora 指示牌 one-hot（5×34）
  510-513 四家立直标志（4）
  514-516 局面：kyoku/honba/tiles_left（3）
  517-520 手牌张数/14, 副露数/4, 轮次/20, 门清（4）
  521-523 trigger_type one-hot（3）
  524 trigger_tile 有效标志（1）
  525-558 trigger_tile one-hot（34）

标签：expert 动作 → 47 槽（用 train_hanchan.slot_of）
"""
import sys, glob, os, io, tarfile, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from houou_data import read_shard, decode_match, ACT_DISCARD, ACT_RIICHI_DISCARD, ACT_CHI, \
    ACT_PON, ACT_KAKAN, ACT_ANKAN, ACT_RON, ACT_TSUMO, ACT_PASS
from train_hanchan import slot_of, N_ACTIONS
from mahjong import env as E


def decision_to_features(d):
    """Decision -> (560 维特征向量, 47 槽标签 or -1)"""
    f = np.zeros(560, dtype=np.float32)  # 必须 float32：uint8 索引 >255 会溢出

    # 0-33: 自己手牌
    f[0:34] = d.hand_counts[:34]

    # 34-67: 自己弃牌
    f[34:68] = d.discards[:34]

    # 68-203: 四家弃牌（4×34=136）
    f[68:204] = d.discards_by[:4, :34].reshape(-1)

    # 204-339: 四家副露（houou_data.Decision.melds_by 是 [(t1,t2,t3),...] 元组列表）
    for i, meld_list in enumerate(d.melds_by[:4]):
        for meld in meld_list:
            for t in meld:
                if 0 <= int(t) < 34:
                    f[204 + i * 34 + int(t)] += 1

    # 340-509: dora 指示牌（前 5 个）
    for i, ind in enumerate(d.dora_indicators[:5]):
        ind = int(ind)  # numpy uint8 下溢会传染整个索引表达式
        if ind > 0:
            base = ind - 1  # houou 是 1-based
            if 0 <= base < 34:
                f[340 + i * 34 + base] = 1

    # 510-513: 四家立直
    f[510:514] = d.riichi_flags[:4]

    # 514-516: 局面
    f[514] = d.kyoku / 4.0
    f[515] = d.honba / 3.0
    f[516] = d.tiles_left / 70.0

    # 517-520: 手牌张数 / 14, 副露数 / 4, 轮次 / 20, 门清
    hand_total = sum(d.hand_counts[:34])
    f[517] = hand_total / 14.0
    f[518] = len(d.melds) / 4.0
    f[519] = min(d.turn, 20) / 20.0
    f[520] = 1.0 if len(d.melds) == 0 else 0.0

    # 521-523: trigger_type one-hot（self/call/other）
    tt = d.trigger_type
    if tt == 0:
        f[521] = 1
    elif tt == 1:
        f[522] = 1
    else:
        f[523] = 1

    # 524: trigger_tile 是否有效（响应类决策）
    if d.trigger_tile >= 0:
        f[524] = 1
        if d.trigger_tile < 34:
            f[525 + d.trigger_tile] = 1  # 525-558: trigger_tile one-hot

    # 标签：expert 动作 → 47 槽
    act_type, tile_base, consumed = d.expert
    # 构造伪 Action 对象给 slot_of
    act_map = {
        ACT_DISCARD: E.ACT_DISCARD, ACT_RIICHI_DISCARD: E.ACT_RIICHI,
        ACT_CHI: E.ACT_CHI, ACT_PON: E.ACT_PON, ACT_KAKAN: E.ACT_ADD_KAN,
        ACT_ANKAN: E.ACT_ANKAN, ACT_RON: E.ACT_RON, ACT_TSUMO: E.ACT_TSUMO,
        ACT_PASS: E.ACT_PASS,
    }
    if act_type not in act_map:
        return f, -1
    e_type = act_map[act_type]
    e_tile = tile_base if tile_base is not None and tile_base >= 0 else -1
    e_arg = consumed[0] if len(consumed) > 0 else -1
    from mahjong.env import Action
    act = Action(e_type, e_tile, e_arg)
    slot = slot_of(act)
    if slot < 0 or slot >= N_ACTIONS:
        return f, -1

    return f, slot


def collect_shards(shard_paths, max_decisions=None):
    """流式采集：逐 npz 解码转特征，内存恒定（不一次性 load 整 shard）。"""
    cap = max_decisions or 200000
    X = np.empty((cap, 560), dtype=np.float32)
    Y = np.empty(cap, dtype=np.int64)
    total = 0
    for path in shard_paths:
        if total >= cap:
            break
        print(f"读取 {os.path.basename(path)}...", flush=True)
        try:
            with tarfile.open(path, 'r') as tf:
                for m in tf.getmembers():
                    if total >= cap:
                        break
                    if not m.name.endswith('.npz'):
                        continue
                    try:
                        data = np.load(io.BytesIO(tf.extractfile(m).read()))
                        decisions = decode_match(data)
                    except Exception as e:
                        if total == 0 and 'dbg' not in globals():
                            globals()['dbg'] = 1
                            print(f"  [debug] decode 失败 {m.name[:30]}: {type(e).__name__}:{str(e)[:80]}", flush=True)
                        continue
                    for d in decisions:
                        if total >= cap:
                            break
                        feat, label = decision_to_features(d)
                        if label < 0:
                            continue
                        X[total] = feat
                        Y[total] = label
                        total += 1
            print(f"  {os.path.basename(path)} 累计 {total}", flush=True)
        except Exception as e:
            print(f"  跳过（{type(e).__name__}:{str(e)[:80]}）", flush=True)
    return X[:total], Y[:total]


class ActionMLP(nn.Module):
    def __init__(self, in_dim=560, hidden=512, n_actions=47):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-glob", default="data/houou/*.tar")
    ap.add_argument("--max-shards", type=int, default=5)
    ap.add_argument("--max-decisions", type=int, default=500000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="checkpoints/tenhou_mlp.pt")
    args = ap.parse_args()

    torch.set_num_threads(4)
    paths = sorted(glob.glob(args.shard_glob))[:args.max_shards]
    print(f"使用 {len(paths)} 个 shard: {[os.path.basename(p) for p in paths]}")

    print("采集数据...")
    X, Y = collect_shards(paths, args.max_decisions)
    print(f"样本 {len(X)} 维度 {X.shape[1]} 标签分布 {np.bincount(Y, minlength=N_ACTIONS)[:10]}...")

    # 划分 train/val
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    nv = max(1, len(X) // 10)
    va_idx, tr_idx = idx[:nv], idx[nv:]
    Xtr, Ytr = X[tr_idx], Y[tr_idx]
    Xva, Yva = X[va_idx], Y[va_idx]

    device = torch.device("cpu")
    model = ActionMLP(in_dim=X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()

    Xtr_t = torch.from_numpy(Xtr)
    Ytr_t = torch.from_numpy(Ytr)
    Xva_t = torch.from_numpy(Xva)
    Yva_t = torch.from_numpy(Yva)

    best_acc = 0.0
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        tot_loss = 0.0
        for i in range(0, len(Xtr_t), args.bs):
            b = perm[i:i + args.bs]
            opt.zero_grad()
            logits = model(Xtr_t[b])
            loss = lossf(logits, Ytr_t[b])
            loss.backward()
            opt.step()
            tot_loss += float(loss) * len(b)

        model.eval()
        with torch.no_grad():
            logits = model(Xva_t)
            pred = logits.argmax(1)
            acc = float((pred == Yva_t).float().mean())
        print(f"epoch {ep} loss={tot_loss/len(Xtr):.4f} val_acc={acc:.4f}", flush=True)
        if acc > best_acc:
            best_acc = acc
            torch.save({"state_dict": model.state_dict(),
                        "in_dim": X.shape[1],
                        "hidden": 512,
                        "n_actions": N_ACTIONS,
                        "best_acc": best_acc}, args.out)

    print(f"best val_acc={best_acc:.4f}, 保存至 {args.out}")


if __name__ == "__main__":
    main()
