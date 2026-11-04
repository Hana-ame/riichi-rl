#!/usr/bin/env python3
"""MahjongNet 模型大小 vs 推理响应时间基准（可复现）。

测什么
------
1. 参数量；
2. CPU 单样本前向延迟：batch=1，warmup 20 后计时；
3. CPU 批量吞吐：batch=64 / batch=200 的前向 ms 与 samples/s；
4. （可选）GPU 单样本延迟，仅参考（项目硬约束是 CPU-only）；
5. 端到端单步决策（编码 + 前向 + 掩码 + argmax），参考 play_model.py 的 play 循环。

方法
----
- `model.eval()` + `torch.no_grad()`；
- 先 warmup 再计时，每档重复 3 轮取均值（池化 3 轮的全部单次读数），
  以降低 WSL 主机上其他进程（serve_game 等）带来的抖动；
- 单次延迟用 `time.perf_counter()` 逐次计时；GPU 用整段计时 + `synchronize()`。

线程数默认 **1**，不是 cpu_only 的 12。原因见 rules/MODEL_BENCH.md：
本机（Ryzen 5 3600 / WSL2）12 线程跑 batch=1 的小卷积时，OpenMP 同步开销
比计算本身还大，实测 12 线程 ~11ms vs 1 线程 ~2.8ms（同机、同权重）。
`--threads default` 可复现 cpu_only 的默认线程数，两套数字都保留。

用法
----
    PY=/mnt/d/WorkPlace/my_rl_project/venv/bin/python
    $PY bench/bench_model.py                      # 主表：全部配置，1 线程
    $PY bench/bench_model.py --threads default     # cpu_only 默认线程数
    $PY bench/bench_model.py --mode threads        # 线程数扫描（1/2/4/6/8/12）
    $PY bench/bench_model.py --gpu                 # GPU 参考（不 import cpu_only）
    $PY bench/bench_model.py --quick               # 冒烟（迭代数缩小）

输出
----
- markdown 表打印到 stdout；
- 原始读数写到 bench/results.json（或 --out-json 指定）。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OBS_DIM = 22 * 34 + 26  # 774：obs2 编码拍平


# --------------------------------------------------------------------------
# 配置表
# --------------------------------------------------------------------------
# 现有权重两档：
#   riichi_net.pt = convformer_v1, channels=160, n_blocks=6, n_actions=54,
#                   extra_scalar=0 -> 1,652,967 参数（任务里的「约 1.65M」）
#   ppo_v1.pt 不存在；用 ckfw.pt 作 ~444K 档的替代（64ch/4blk，453,232 参数，
#   但它是 hanchan_v1：n_actions=47, extra_scalar=12）。表格里会注明。
CONFIGS = [
    # 444,791 参数，正好是任务里说的「约 444K」；ppo_v1.pt 文件不存在，
    # 所以这一档是「按同结构重建」而不是载入权重。
    dict(label="64x4", channels=64, n_blocks=4,
         note="≈444K；与缺失的 ppo_v1.pt 同尺寸（无权重可载）"),
    # 现存最小的完整权重。ckfw.pt 是 hanchan_v1：47 动作、touch_dim=8、
    # extra_scalar=12（head_fc 输入 5*64+26+8+12=366），共 453,232 参数。
    dict(label="ckpt-64x4-ckfw", channels=64, n_blocks=4, touch_dim=8,
         extra_scalar=12, n_actions=47, ckpt="ckfw.pt",
         note="现有权重 ckfw.pt（hh_v1 头：47动作/touch8/extra12）"),
    dict(label="96x4", channels=96, n_blocks=4),
    dict(label="96x6", channels=96, n_blocks=6),
    dict(label="128x6", channels=128, n_blocks=6),
    dict(label="128x8", channels=128, n_blocks=8),
    dict(label="160x4", channels=160, n_blocks=4),
    dict(label="ckpt-160x6", channels=160, n_blocks=6,
         ckpt="checkpoints/riichi_net.pt",
         note="现有权重 riichi_net.pt"),
    dict(label="160x8", channels=160, n_blocks=8),
    dict(label="256x4", channels=256, n_blocks=4),
    dict(label="256x6", channels=256, n_blocks=6),
]

# 端到端只选代表配置：最小 / 中 / 现有权重 / 最大（均为 54 动作、无额外标量）
E2E_LABELS = ("64x4", "96x6", "ckpt-160x6", "256x6")


def obs_dim(cfg):
    """该配置期望的输入维度：774 + touch_dim + extra_scalar。"""
    return OBS_DIM + cfg.get("touch_dim", 0) + cfg.get("extra_scalar", 0)


def parse_args():
    ap = argparse.ArgumentParser(description="MahjongNet 推理基准")
    ap.add_argument("--mode", choices=("sizes", "threads", "all"),
                    default="sizes")
    ap.add_argument("--threads", default="1",
                    help="整数线程数，或 'default' 表示沿用 cpu_only 的默认")
    ap.add_argument("--gpu", action="store_true",
                    help="在 CUDA 上测（不 import cpu_only；仅参考）")
    ap.add_argument("--single-iters", type=int, default=200)
    ap.add_argument("--b64-iters", type=int, default=30)
    ap.add_argument("--b200-iters", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--quick", action="store_true",
                    help="缩小迭代数，仅用于冒烟")
    ap.add_argument("--only", default=None,
                    help="逗号分隔的配置 label，只测这些")
    ap.add_argument("--out-json", default=os.path.join("bench", "results.json"))
    return ap.parse_args()


# --------------------------------------------------------------------------
# 环境
# --------------------------------------------------------------------------
def init_env(args):
    """选设备、设线程。import torch 的位置很关键（cpu_only 必须在 torch 前）。"""
    if args.gpu:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch, device
    import cpu_only  # noqa: F401  # 设 CUDA_VISIBLE_DEVICES=""，并设默认线程数
    import torch
    if args.threads != "default":
        torch.set_num_threads(int(args.threads))
    return torch, "cpu"


def env_meta(torch, device, args):
    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "device": device,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "os_cpu_count": os.cpu_count(),
        "cpu": platform.processor() or "unknown",
        "loadavg": list(os.getloadavg()),
        "args": {k: v for k, v in vars(args).items()},
    }
    if device == "cuda":
        meta["gpu"] = torch.cuda.get_device_name(0)
        meta["gpu_capability"] = list(torch.cuda.get_device_capability(0))
    return meta


# --------------------------------------------------------------------------
# 计时
# --------------------------------------------------------------------------
def _stats(ms_list):
    xs = sorted(ms_list)
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    return {
        "mean": mean,
        "std": var ** 0.5,
        "p50": xs[n // 2],
        "p90": xs[min(n - 1, int(n * 0.9))],
        "min": xs[0],
        "max": xs[-1],
        "n": n,
    }


def measure(torch, fn, warmup, iters, repeats, device):
    """先 warmup，再测 repeats 轮；返回 ms 统计（池化所有读数）。

    CPU 逐次计时（保留分布）；GPU 每轮整段计时后除以 iters（避免 sync 抖动）。
    """
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    pool = []
    for _ in range(repeats):
        if device == "cuda":
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
            pool.append((time.perf_counter() - t0) / iters * 1000.0)
        else:
            for _ in range(iters):
                t0 = time.perf_counter()
                fn()
                pool.append((time.perf_counter() - t0) * 1000.0)
    return _stats(pool)


# --------------------------------------------------------------------------
# 模型
# --------------------------------------------------------------------------
def build_model(torch, cfg, device):
    from mahjong.net import MahjongNet, count_params

    model = MahjongNet(
        channels=cfg["channels"],
        n_blocks=cfg["n_blocks"],
        n_actions=cfg.get("n_actions", 54),
        touch_dim=cfg.get("touch_dim", 0),
        extra_scalar=cfg.get("extra_scalar", 0),
    )
    loaded = None
    ckpt = cfg.get("ckpt")
    if ckpt:
        path = os.path.join(ROOT, ckpt)
        if os.path.exists(path):
            ck = torch.load(path, map_location="cpu", weights_only=False)
            sd = ck.get("state_dict", ck)
            try:
                model.load_state_dict(sd, strict=True)
                loaded = ckpt
            except RuntimeError as exc:
                loaded = "load-failed:" + str(exc).splitlines()[0]
        else:
            loaded = "missing:" + ckpt
    model = model.to(device).eval()
    return model, count_params(model), loaded, obs_dim(cfg)


def bench_config(torch, cfg, args, device):
    model, n_params, loaded, dim = build_model(torch, cfg, device)
    x1 = torch.randn(1, dim, device=device)
    x64 = torch.randn(64, dim, device=device)
    x200 = torch.randn(200, dim, device=device)

    with torch.no_grad():
        single = measure(torch, lambda: model(x1), 20, args.single_iters,
                         args.repeats, device)
        b64 = measure(torch, lambda: model(x64), 10, args.b64_iters,
                      args.repeats, device)
        b200 = measure(torch, lambda: model(x200), 5, args.b200_iters,
                       args.repeats, device)

    res = {
        "label": cfg["label"],
        "channels": cfg["channels"],
        "n_blocks": cfg["n_blocks"],
        "extra_scalar": cfg.get("extra_scalar", 0),
        "n_actions": cfg.get("n_actions", 54),
        "params": n_params,
        "loaded": loaded,
        "single_ms": single,
        "batch64_ms": b64,
        "batch200_ms": b200,
        "b64_sps": 64 * 1000.0 / b64["mean"],
        "b200_sps": 200 * 1000.0 / b200["mean"],
        "note": cfg.get("note", ""),
    }
    del model, x1, x64, x200
    if device == "cuda":
        torch.cuda.empty_cache()
    return res


# --------------------------------------------------------------------------
# 端到端单步决策（编码 + 前向 + 掩码 + argmax）
# --------------------------------------------------------------------------
def build_e2e_state(torch):
    import numpy as np
    from mahjong import env as E
    from mahjong import obs2 as O2
    from train_reinforce import N_ACTIONS, slot_of

    env = E.MahjongEnv(seed=12345, dealer=1)
    env.start_turn()
    pending = env.pending_players()
    p = pending[0]
    legal = env.legal_actions(p)
    if not legal:
        raise RuntimeError("seed=12345 的首个 pending 玩家无合法动作")
    encoder = O2.ObsEncoder()
    slots = [slot_of(a) for a in legal]
    mask = torch.full((N_ACTIONS,), -1e9, dtype=torch.float32)
    mask[slots] = 0.0
    return env, p, encoder, slots, mask, np


def bench_e2e(torch, model, args, device):
    env, p, encoder, slots, mask, np = build_e2e_state(torch)
    mask = mask.to(device)

    def encode_vec():
        planes, scalar = encoder.encode(env, p)
        return np.concatenate([planes.ravel(), scalar]).astype(np.float32)

    def forward_only():
        vec = encode_vec()
        x = torch.from_numpy(vec).unsqueeze(0).to(device)
        logits = model(x)[0]
        pick = int((logits + mask).argmax())
        return pick

    with torch.no_grad():
        enc = measure(torch, lambda: encoder.encode(env, p), 20, 100,
                      args.repeats, device)
        e2e = measure(torch, forward_only, 20, 100, args.repeats, device)
    return {"encoder_ms": enc, "e2e_ms": e2e, "legal_slots": slots}


# --------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------
def fmt_ms(stats, nd=2):
    return f"{stats['mean']:.{nd}f}"


def print_table(results, threads_label=""):
    head = ("| 配置 | 参数量 | CPU单次ms | CPU batch64 ms | CPU batch200 ms "
            "| 吞吐 samples/s | 备注 |")
    sep = "|---|---|---|---|---|---|---|"
    print(head)
    print(sep)
    for r in results:
        note = r["note"]
        if r.get("e2e_ms"):
            note = (note + "；" if note else "") + \
                f"端到端 {r['e2e_ms']['mean']:.2f} ms"
        print(f"| {r['label']} | {r['params']:,} | {fmt_ms(r['single_ms'])} "
              f"| {fmt_ms(r['batch64_ms'])} | {fmt_ms(r['batch200_ms'], 1)} "
              f"| {r['b200_sps']:.0f} | {note} |")
    print()


def print_thread_table(rows):
    print("| 线程数 | 配置 | 单次 ms | 单次 p50 | batch200 ms | batch200 samples/s |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['threads']} | {r['label']} | {r['single_ms']['mean']:.2f} "
              f"| {r['single_ms']['p50']:.2f} | {r['batch200_ms']['mean']:.1f} "
              f"| {r['b200_sps']:.0f} |")
    print()


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    args = parse_args()
    if args.quick:
        args.single_iters = 40
        args.b64_iters = 8
        args.b200_iters = 5
        args.repeats = 1

    torch, device = init_env(args)
    meta = env_meta(torch, device, args)
    only = set(args.only.split(",")) if args.only else None

    payload = {"meta": meta, "results": [], "thread_scan": []}
    t_start = time.time()

    if args.mode in ("sizes", "all"):
        results = []
        for cfg in CONFIGS:
            if only and cfg["label"] not in only:
                continue
            print(f"[bench] {cfg['label']} ...", file=sys.stderr, flush=True)
            r = bench_config(torch, cfg, args, device)
            results.append(r)
            print(f"[bench]   params={r['params']:,} "
                  f"single={r['single_ms']['mean']:.2f}ms "
                  f"b200={r['batch200_ms']['mean']:.1f}ms", file=sys.stderr,
                  flush=True)

        if args.mode == "sizes":
            # 端到端：只测 CPU 代表配置（GPU 端到端另说）
            if device == "cpu":
                for r in results:
                    if r["label"] in E2E_LABELS:
                        cfg = next(c for c in CONFIGS if c["label"] == r["label"])
                        model, _, _, _ = build_model(torch, cfg, device)
                        e2e = bench_e2e(torch, model, args, device)
                        r["e2e_ms"] = e2e["e2e_ms"]
                        r["encoder_ms"] = e2e["encoder_ms"]
                        del model
        payload["results"] = results
        if device == "cpu":
            print_table(results)
        else:
            print_table(results)

    if args.mode in ("threads", "all") and device == "cpu":
        scan_labels = ("64x4", "ckpt-160x6")
        cfg_map = {c["label"]: c for c in CONFIGS}
        for nt in (1, 2, 4, 6, 8, 12):
            torch.set_num_threads(nt)
            for label in scan_labels:
                cfg = cfg_map[label]
                model, n_params, _, dim = build_model(torch, cfg, "cpu")
                x1 = torch.randn(1, dim)
                x200 = torch.randn(200, dim)
                s_iters, s_rep = (30, 1) if args.quick else (100, 2)
                b_iters, b_rep = (5, 1) if args.quick else (10, 2)
                with torch.no_grad():
                    sm = measure(torch, lambda: model(x1), 10, s_iters, s_rep,
                                 "cpu")
                    bm = measure(torch, lambda: model(x200), 3, b_iters,
                                 b_rep, "cpu")
                payload["thread_scan"].append({
                    "threads": nt, "label": label, "params": n_params,
                    "single_ms": sm, "batch200_ms": bm,
                    "b200_sps": 200 * 1000.0 / bm["mean"],
                })
                print(f"[threads] nt={nt} {label}: "
                      f"single={sm['mean']:.2f}ms b200={bm['mean']:.1f}ms",
                      file=sys.stderr, flush=True)
        print_thread_table(payload["thread_scan"])

    payload["meta"]["elapsed_s"] = round(time.time() - t_start, 1)
    payload["meta"]["loadavg_after"] = list(os.getloadavg())

    out = args.out_json
    if not os.path.isabs(out):
        out = os.path.join(ROOT, out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"[bench] wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()