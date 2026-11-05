"""版本天梯：所有版本同池混战 + 「后版本必须强过前版本才过」的晋级闸门。

用户要求（2026-11-05）：「新版本 V1/V2/... 都在一起比；靠后的版本需要比靠前的
版本强才能过。」

规则：
1. 所有版本在**同一个池**里随机组 4 人局（同一批牌山、座位随机），算 Elo；
2. 对**相邻版本** (V(k-1), V(k)) 统计同局净点数差（后 - 前）的均值与 95% CI；
3. 晋级判定：
   - PASS：后版本 Elo >= 前版本，且同局净点差 95% CI 下界 > 0（统计显著更强）；
   - FAIL：点差均值 <= 0，或 Elo 更低；
   - 证据不足（HOLD）：CI 跨 0 且样本不够——不算过，继续加局。

用法：
    python version_ladder.py --lineage teacher47 --games 600
    python version_ladder.py --ckpts "v1.pt,v2.pt,v3.pt" --games 600
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cpu_only  # noqa: F401  —— 必须先于 torch 锁 CPU-only
import torch

from arena_elo import (HanchanAgent, V1Agent, BCAgent, play_one,
                       elo_update, CKPT)
from cpu_only import set_inference_threads

ROOT = os.path.dirname(os.path.abspath(__file__))


def detect_arch(path):
    """看 ckpt 的策略头输出维度判断架构：47 半庄 / 54 convformer / 243 BC。"""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    for key in ("policy.weight", "policy.bias"):
        if key in sd:
            n = int(sd[key].shape[0])
            return {47: "hanchan47", 54: "v1_54", 243: "bc_243"}.get(n, "hanchan47")
    return "hanchan47"


def _make_agent(arch, name, path):
    if arch == "auto":
        arch = detect_arch(path)
    if arch == "hanchan47":
        return HanchanAgent(name, path)
    if arch == "v1_54":
        return V1Agent(name, path)
    if arch == "bc_243":
        return BCAgent(name, path)
    raise ValueError(arch)


def build_lineage(lineage, ckpts_arg, arch):
    """返回 [(版本名, ckpt路径)]，按版本先后排序。"""
    if ckpts_arg:
        paths = [p.strip() for p in ckpts_arg.split(",") if p.strip()]
        out = []
        for p in paths:
            if not os.path.isabs(p):
                base = os.path.basename(p)
                if not base.endswith(".pt"):
                    base += ".pt"
                p = os.path.join(CKPT, base)
            if not os.path.exists(p):
                print(f"[skip] 不存在 {p}", flush=True)
                continue
            out.append((os.path.basename(p).replace(".pt", ""), p))
        return out

    if lineage == "teacher47":
        names = ["teacher47_v1", "teacher47_v2", "teacher47_v3",
                 "style_rl_v1"]
    elif lineage == "from_games":
        names = ["from_games_v1", "from_games_v2"]
    elif lineage == "a2c":
        # 按 iter 递增的早期 RL 版本（同架构 47 槽）
        # 只选 47 槽版本（r2/r3/r8 是 54 槽，架构不同，混进来会 size mismatch）
        names = ["a2c_r1_it10", "a2c_r1_it20", "a2c_r1_it30",
                 "a2c_r4_it30", "a2c_r9_it25", "a2c_r10_it25", "a2c_r11_it5"]
    else:
        raise ValueError(f"未知血缘 {lineage}")

    pattern = os.path.join(CKPT, "*.pt")
    existing = {os.path.basename(p).replace(".pt", ""): p
                for p in glob.glob(pattern)}
    out = []
    for n in names:
        if n in existing:
            out.append((n, existing[n]))
        else:
            print(f"[skip] {n}.pt 不存在", flush=True)
    return out


def ci95(xs):
    n = len(xs)
    if n < 2:
        return 0.0, 0.0, 0.0
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    se = math.sqrt(var / n)
    return m, m - 1.96 * se, m + 1.96 * se


def verdict_of(diffs, elo_delta, band=60.0, min_n=20):
    """晋级判定：PASS / FAIL / HOLD。

    PASS：净点差 95% CI 下界 > 0（显著更强），且 Elo 没有显著下滑
    （elo_delta >= -band；band ≈ 2×Elo 标准误）。
    FAIL：净点差显著为负（CI 上界 < 0）或 Elo 显著下滑（elo_delta < -band）。
    HOLD：其余（点差跨 0 / 样本不足 / 点差与 Elo 结论矛盾）——不算过。
    """
    m, lo, hi = ci95(diffs)
    if len(diffs) < min_n:
        return "HOLD", m, lo, hi
    elo_bad = elo_delta < -band
    if m > 0 and lo > 0 and not elo_bad:
        return "PASS", m, lo, hi
    if (m < 0 and hi < 0) or elo_bad:
        return "FAIL", m, lo, hi
    return "HOLD", m, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineage", default="teacher47",
                    choices=["teacher47", "from_games", "a2c"])
    ap.add_argument("--ckpts", default="",
                    help="逗号分隔的 ckpt 列表，给出时忽略 --lineage")
    ap.add_argument("--arch", default="hanchan47",
                    choices=["hanchan47", "v1_54", "bc_243", "auto"],
                    help="auto=按 ckpt 策略头维度自动识别（混池用）")
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    set_inference_threads(2)
    lineage = build_lineage(args.lineage, args.ckpts, args.arch)
    if len(lineage) < 2:
        print("版本不足 2 个，无法做天梯。", flush=True)
        return 1
    agents = [_make_agent(args.arch, name, path) for name, path in lineage]
    names = [a.name for a in agents]
    print(f"版本池 {len(agents)} 个：{names}", flush=True)

    rng = random.Random(args.seed)
    ratings = {i: 1500.0 for i in range(len(agents))}
    games_played = {i: 0 for i in range(len(agents))}
    wins = {i: 0 for i in range(len(agents))}
    pt_sum = {i: 0.0 for i in range(len(agents))}
    pair_diffs = {(i, j): [] for i in range(len(agents))
                  for j in range(len(agents)) if i != j}
    t0 = time.time()

    for g in range(args.games):
        # 版本池不足 4 个时按循环补齐座位（2 版本 → 各 2 席），保证 4 人局
        base = list(range(len(agents)))
        while len(base) < 4:
            base = base + list(range(len(agents)))
        sel = rng.sample(base, 4)
        picked = [agents[i] for i in sel]
        out = play_one(picked, seed=args.seed * 7919 + g * 131)
        if out is None:
            continue
        delta = out[0]
        winner = out[1]
        elo_update(ratings, sel, delta)
        for pos, i in enumerate(sel):
            games_played[i] += 1
            pt_sum[i] += delta[pos]
            if winner == pos:
                wins[i] += 1
            for pos2, j in enumerate(sel):
                if i == j:
                    continue
                pair_diffs[(j, i)].append(delta[pos] - delta[pos2])
        if (g + 1) % 100 == 0:
            print(f"[ladder] {g+1}/{args.games} ({time.time()-t0:.0f}s)",
                  flush=True)

    print()
    print(f"版本天梯：{args.games} 局，同池随机 4 人局（2 线程）")
    print(f"{'版本':<22}{'Elo':>8}{'局数':>7}{'和牌率':>8}{'平均点':>9}")
    print("-" * 56)
    for i, name in enumerate(names):
        g = max(1, games_played[i])
        print(f"{name:<22}{ratings[i]:>8.1f}{games_played[i]:>7}"
              f"{100.0*wins[i]/g:>7.1f}%{pt_sum[i]/g:>+9.0f}")

    print()
    print("晋级闸门（后版本必须强过前版本）：")
    print(f"{'对比':<42}{'净点差(后-前)':>14}{'95% CI':>22}{'判定':>10}")
    print("-" * 90)
    gates = []
    for k in range(1, len(agents)):
        prev, cur = k - 1, k
        diffs = pair_diffs[(prev, cur)]      # cur - prev
        elo_delta = ratings[cur] - ratings[prev]
        band = 2.0 * (400.0 / max(1.0, sum(games_played.values())
                                  / max(1, len(agents))) ** 0.5)
        verdict, m, lo, hi = verdict_of(diffs, elo_delta, band)
        gates.append({
            "prev": names[prev], "cur": names[cur], "n": len(diffs),
            "mean_diff": round(m, 1), "ci_lo": round(lo, 1),
            "ci_hi": round(hi, 1),
            "elo_delta": round(elo_delta, 1), "elo_band": round(band, 1),
            "verdict": verdict,
        })
        print(f"{names[prev]+' -> '+names[cur]:<42}{m:>+14.1f}"
              f"  [{lo:>+8.0f}, {hi:>+8.0f}] n={len(diffs):<5}{verdict:>10}")

    passed = sum(1 for g in gates if g["verdict"] == "PASS")
    print()
    print(f"结论：{passed}/{len(gates)} 个相邻版本晋级 PASS；"
          f"FAIL {sum(1 for g in gates if g['verdict']=='FAIL')}，"
          f"HOLD {sum(1 for g in gates if g['verdict']=='HOLD')}")

    if args.out:
        payload = {"lineage": args.lineage, "arch": args.arch,
                   "games": args.games,
                   "versions": [{"name": names[i],
                                 "elo": round(ratings[i], 1),
                                 "games": games_played[i],
                                 "win_rate": round(100.0 * wins[i]
                                                   / max(1, games_played[i]), 1),
                                 "avg_pt": round(pt_sum[i]
                                                 / max(1, games_played[i]), 1)}
                                for i in range(len(agents))],
                   "gates": gates}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        print(f"写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())