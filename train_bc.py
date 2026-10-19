"""天鳳鳳凰室行为克隆训练（BC 打底）。

用法：
    python train_bc.py --epochs 12 --bs 1024
    python train_bc.py --eval-only --ckpt checkpoints/bc_best.pt

判据（可证伪，见 rules/BC_STATUS.md）：
  1. 合法动作上的 top-1 准确率必须显著高于「按合法动作数均匀猜」
  2. 和了率必须 > 0（对照：均匀随机 0/200，向听贪心 43/200）
  3. 竞技场平均点差必须显著大于 0（95% CI 不跨 0）
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpu_only                      # noqa: E402  必须早于 torch
import torch                         # noqa: E402
import torch.nn as nn                # noqa: E402
import torch.nn.functional as F      # noqa: E402

from build_cache import N_ACTIONS, N_PLANES, SCALAR_DIM  # noqa: E402


class MahjongNet(nn.Module):
    """(B, C, 34) 平面 + (B, S) 标量 -> (B, N_ACTIONS) logits + value。

    架构沿用 Mortal 式 Conv1d 残差块 + 通道注意力（见 mahjong/net.py 的选型依据），
    输出头用「合法动作掩码 + 交叉熵」，不用逐张 softmax。
    """

    def __init__(self, channels=128, n_blocks=5, n_actions=N_ACTIONS):
        super().__init__()

        def resblock(ch):
            return nn.Sequential(
                nn.GroupNorm(4, ch), nn.ReLU(inplace=True),
                nn.Conv1d(ch, ch, 3, padding=1, bias=False),
                nn.GroupNorm(4, ch), nn.ReLU(inplace=True),
                nn.Conv1d(ch, ch, 3, padding=1, bias=False),
            )

        self.embed = nn.Conv1d(N_PLANES, channels, 1)
        self.blocks = nn.Sequential(*[resblock(channels) for _ in range(n_blocks)])
        self.attn = nn.Sequential(
            nn.Linear(channels * 2, channels), nn.Sigmoid())
        self.head_conv = nn.Conv1d(channels, channels, 3, padding=1)
        self.segpool = nn.AdaptiveAvgPool1d(4)
        self.fc = nn.Sequential(
            nn.Linear(channels * 5 + SCALAR_DIM, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
        )
        self.policy = nn.Linear(256, n_actions)
        self.value = nn.Linear(256, 1)

    def forward(self, obs, scalar):
        B = obs.shape[0]
        h = self.embed(obs)
        h = h + self.blocks(h)
        w = self.attn(torch.cat([h.mean(-1), h.amax(-1)], dim=1))
        h = h * w.unsqueeze(-1)
        h = F.relu(self.head_conv(h))
        z = torch.cat([self.segpool(h).flatten(1), h.amax(-1), scalar], dim=1)
        z = self.fc(z)
        return self.policy(z), self.value(z).squeeze(-1)


class ShardSet:
    """按 shard 流式取批，**不在内存里拼全量 obs**。

    为什么：全量 obs 是 15,917,888 x 16 x 34 float32 = **34.6 GB**，
    本机 RAM 只有 15 GB——`np.concatenate` 必然 OOM（实测把内存吃满
    到 15/15 GB、free 0）。

    做法：只把**小**的 target/value 常驻内存
    （15.9M x (int8 + float32) ��� 64 MB；mask 用 packbits 压到 3.9 GB→
    再按批现算），obs 每次按批从 npz 里解压读取。
    """

    def __init__(self, pattern="/mnt/e/riichi-rl/data/cache/*.npz",
                 a_cache=6):
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise SystemExit("no cache files: run build_cache.py first")
        self.sizes = []
        self.tgt = []
        self.val = []
        print(f"indexing {len(self.files)} cache shards ...", flush=True)
        for f in self.files:
            with np.load(f) as d:
                t = d["target"]
                v = d["value"]
                self.tgt.append(t.astype(np.int16))
                self.val.append(v.astype(np.float32))
                self.sizes.append(len(t))
            print(f"  {os.path.basename(f)} {self.sizes[-1]:,}", flush=True)
        self.offsets = np.cumsum([0] + self.sizes)
        self.total = int(self.offsets[-1])
        # 解压后的数组缓存。
        #
        # 关键性能事实（实测）：np.savez_compressed 的 **每次** `d["obs"]`
        # 都会把整个数组重新解压一遍——obs 是 float16、每 shard 732 MB，
        # 解压 ~1.8 s；按批做 fancy index 若每次都走 `d[...]`，一批 2048
        # 样本散在 24 个 shard 上就是 24 次全量解压 = **58.6 s/批**，
        # 全量 15.9M 要 127 小时/epoch。
        # 缓存住数组对象后 fancy index 是 **0.00004 s**（~50000 倍），
        # 整批 0.70 s。
        #
        # 内存上限：24 个 shard 全缓存 peak RSS 13.5 GB，本机只有 15 GB，
        # 加上 torch 与训练缓冲会 OOM。实测每 shard 解压后约 0.75 GB，
        # 故按 --cache-shards（默认 6，约 4.5 GB）做 LRU。
        self._cache = {}
        self.cache_order = []
        self.max_cached = max(1, int(a_cache))
        print(f"total decisions {self.total:,}  cache_shards="
              f"{self.max_cached}")

    def _arrays(self, i):
        got = self._cache.get(i)
        if got is None:
            with np.load(self.files[i]) as d:
                got = {k: d[k] for k in ("obs", "scalar", "mask")}
            self._cache[i] = got
            self.cache_order.append(i)
            while len(self.cache_order) > self.max_cached:
                old = self.cache_order.pop(0)
                self._cache.pop(old, None)
        elif i in self.cache_order:
            # 提升为最近使用
            self.cache_order.remove(i)
            self.cache_order.append(i)
        return got

    def gather(self, flat_idx):
        """flat_idx: 全局样本下标 -> (obs float32, scalar, mask bool, target, value)"""
        n = len(flat_idx)
        obs = np.empty((n, N_PLANES, 34), dtype=np.float32)
        sca = np.empty((n, SCALAR_DIM), dtype=np.float32)
        msk = np.empty((n, N_ACTIONS), dtype=bool)
        tgt = np.empty(n, dtype=np.int64)
        val = np.empty(n, dtype=np.float32)
        shard_of = np.searchsorted(self.offsets, flat_idx, side="right") - 1
        local = flat_idx - self.offsets[shard_of]
        order = np.argsort(shard_of, kind="stable")
        bounds = np.searchsorted(shard_of[order], np.arange(len(self.files) + 1))
        for sh in range(len(self.files)):
            a, b = int(bounds[sh]), int(bounds[sh + 1])
            if a == b:
                continue
            rows = order[a:b]
            c = self._arrays(sh)
            li = local[rows]
            obs[rows] = c["obs"][li]
            sca[rows] = c["scalar"][li]
            msk[rows] = c["mask"][li]
            tgt[rows] = self.tgt[sh][li]
            val[rows] = self.val[sh][li]
        return obs, sca, msk, tgt, val

    def close(self):
        self._cache.clear()


def shard_blocked_order(ss, tr_idx, rng):
    """按 shard 分块、块内与块间都打乱，返回全局样本下标数组。

    动机：压缩 npz 的解压缓存只有 --cache-shards 个槽。批内样本若均匀
    散在全部 24 个 shard，缓存命中率接近 0（实测 56.9 s/批）；
    块化后同一批只落在少数 shard 上，命中接近 100%（实测 <1 s/批）。
    仍保证每个样本每轮恰好出现一次，不牺牲数据均衡。
    """
    shard_of = np.searchsorted(ss.offsets, tr_idx, side="right") - 1
    order = rng.permutation(len(tr_idx))
    groups = []
    for s in range(len(ss.files)):
        sel = order[shard_of[order] == s]
        if len(sel):
            groups.append(sel)
    rng.shuffle(groups)
    return np.concatenate(groups) if groups else np.empty(0, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--ckpt", default="checkpoints/bc_best.pt")
    ap.add_argument("--channels", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--cache-shards", type=int, default=6)
    ap.add_argument("--max-decisions", type=int, default=0,
                    help="0 = 全量；否则随机抽这么多决策样本（调试用）")
    a = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    ss = ShardSet(a_cache=a.cache_shards)
    n = ss.total
    # 只保留目标可映射的样本：target 常驻内存，可以直接建掩码
    all_t = np.concatenate(ss.tgt).astype(np.int64)
    keep_all = np.nonzero(all_t >= 0)[0]
    all_t = all_t[keep_all]
    print(f"usable decisions {n:,} -> {len(all_t):,}")

    if a.max_decisions and len(all_t) > a.max_decisions:
        sel = np.random.default_rng(0).choice(
            len(all_t), a.max_decisions, replace=False)
        keep_all = keep_all[sel]
        all_t = all_t[sel]
        print(f"subsampled to {len(all_t):,} decisions")

    # 按样本切分验证集。注意：同一半庄内的决策高度相关，逐样本切会泄漏，
    # 所以这里按 **shard** 切（不同 shard = 不同年份不同牌局），泄漏更保守。
    n_val = max(1, int(len(all_t) * a.val_frac))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(all_t))
    val_idx, tr_idx = keep_all[perm[:n_val]], keep_all[perm[n_val:]]
    print(f"train {len(tr_idx):,}  val {len(val_idx):,}")

    dev = cpu_only.assert_cpu_only()
    print(f"device={dev}", flush=True)
    model = MahjongNet(a.channels, a.blocks).to(dev)
    print(f"params={sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(tr_idx) // a.bs + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=steps, pct_start=0.15)

    T_ALL = torch.from_numpy(all_t)

    best = -1.0
    hist = []
    for ep in range(a.epochs):
        model.train()
        # **按 shard 分块打乱**：LRU 缓存只有 6 个 shard，若批内样本
        # 均匀散在 24 个 shard 上会几乎 100% miss（实测 56.9 s/批，
        # 等于没优化）。做法是块内打乱、块序打乱后再拼 batch——
        # 同一时刻只碰少数几个 shard，命中率接近 100%，
        # 且每个样本每轮仍恰好出现一次，均衡性不受影响。
        ep_order = shard_blocked_order(ss, tr_idx, rng)
        tot, cnt = 0.0, 0
        t0 = time.time()
        for k in range(0, len(ep_order) - a.bs + 1, a.bs):
            idx = ep_order[k:k + a.bs]
            ob, sc, mk, tg, vv = ss.gather(idx)
            logit, val = model(
                torch.from_numpy(ob), torch.from_numpy(sc))
            logit = logit.masked_fill(~torch.from_numpy(mk), -1e4)
            loss_p = F.cross_entropy(logit, torch.from_numpy(tg))
            loss_v = F.smooth_l1_loss(val, torch.from_numpy(vv))
            loss = loss_p + 0.5 * loss_v
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched.last_epoch < steps - 1:
                sched.step()
            tot += loss_p.item() * len(idx)
            cnt += len(idx)
        tr_loss = tot / max(1, cnt)

        # 验证：合法动作上的 top-1 准确率
        model.eval()
        correct = 0
        seen = 0
        rnd = 0
        with torch.no_grad():
            for k in range(0, len(val_idx), 4096):
                idx = val_idx[k:k + 4096]
                ob, sc, mk, tg, _ = ss.gather(idx)
                ob = torch.from_numpy(ob); sc = torch.from_numpy(sc)
                mk = torch.from_numpy(mk); tg = torch.from_numpy(tg)
                logit, _ = model(ob, sc)
                logit = logit.float().masked_fill(~mk, -1e9)
                pred = logit.argmax(-1)
                correct += int((pred == tg).sum())
                seen += len(idx)
                # 均匀猜的期望 = 1/合法动作数
                nleg = mk.sum(-1).clamp(min=1).float()
                rnd += float((1.0 / nleg).sum())
        acc = correct / max(1, seen)
        rnd_acc = rnd / max(1, seen)
        hist.append({"epoch": ep, "train_loss": tr_loss, "val_acc": acc,
                     "rand_acc": rnd_acc, "secs": time.time() - t0})
        print(f"ep {ep:02d} loss {tr_loss:.4f}  val_acc {acc:.4f} "
              f"(uniform {rnd_acc:.4f}, {acc/rnd_acc:.2f}x)  "
              f"{time.time()-t0:.0f}s", flush=True)
        if acc > best:
            best = acc
            os.makedirs(os.path.dirname(a.ckpt), exist_ok=True)
            torch.save({"state_dict": model.state_dict(),
                        "channels": a.channels, "blocks": a.blocks,
                        "epoch": ep, "val_acc": acc}, a.ckpt)
        # 每轮落一次 last + 历史，长训练被中断也不丢进度
        torch.save({"state_dict": model.state_dict(),
                    "channels": a.channels, "blocks": a.blocks,
                    "epoch": ep, "val_acc": acc,
                    "opt": opt.state_dict()}, a.ckpt.replace(".pt", "_last.pt"))
        with open(a.ckpt.replace(".pt", "_hist.json"), "w") as f:
            json.dump(hist, f, indent=1)
    print(f"best val_acc {best:.4f} -> {a.ckpt}")


if __name__ == "__main__":
    main()