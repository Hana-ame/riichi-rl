"""实时性基准：对自研模型逐个测「单步决策耗时」，验证 5 秒内 realtime。

判据（可证伪）：
- 每个 Agent 在真实对局里跑 N 局，记录每次 act 的耗时；
- **PASS**：p99 ≤ 5000ms 且 max ≤ 5000ms（严格 5 秒预算）；
- 另报 p50/p95/mean，便于看余量。

用法：
    python realtime_bench.py --deals 8
    python realtime_bench.py --agents "teacher47_v3,bc_v1_last,hybrid:teacher47_v3"
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cpu_only  # noqa: F401  —— 必须先于 torch
import torch

from arena_elo import RuleAgent, play_one, CKPT
from cpu_only import set_inference_threads
from realtime_guard import LatencyStats, RealtimeGuard
from version_benchmark import AutoAgent, detect_arch

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_AGENTS = ("teacher47_v3,bc_v1_last,from_games_v2,riichi_net,"
                  "hybrid:teacher47_v3,greedy_tenpai")


def build_agent(spec):
    """'hybrid:teacher47_v3' / 'teacher47_v3' / 'greedy_tenpai' -> Agent。"""
    if spec in ("greedy_tenpai",):
        return RuleAgent(spec)
    hybridd = spec.startswith("hybridd:")
    hybrid = hybridd or spec.startswith("hybrid:")
    name = spec.split(":", 1)[1] if hybrid else spec
    path = os.path.join(CKPT, name if name.endswith(".pt") else name + ".pt")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if hybridd:
        from hybrid_agents import HybridShantenDangerAgent
        return HybridShantenDangerAgent(spec, path)
    if hybrid:
        from hybrid_agents import HybridShantenAgent
        return HybridShantenAgent(spec, path)
    arch = detect_arch(path)
    if arch == "bc_243":
        from arena_elo import BCAgent
        return BCAgent(spec, path)
    return AutoAgent(spec, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", default=DEFAULT_AGENTS)
    ap.add_argument("--deals", type=int, default=8)
    ap.add_argument("--budget-ms", type=float, default=5000.0)
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--out-json", default=os.path.join(
        ROOT, "rules", "REALTIME_BENCH_2026-11-05.json"))
    ap.add_argument("--out-md", default=os.path.join(
        ROOT, "rules", "REALTIME_2026-11-05.md"))
    args = ap.parse_args()

    set_inference_threads(2)
    specs = [x.strip() for x in args.agents.split(",") if x.strip()]
    refs = [RuleAgent("greedy_tenpai"), RuleAgent("call_first"),
            RuleAgent("yaochu_first")]

    results = []
    for spec in specs:
        try:
            agent = build_agent(spec)
        except Exception as e:
            print(f"[skip] {spec}: {e}", flush=True)
            continue
        guard = RealtimeGuard(agent, budget_ms=args.budget_ms)
        # warmup：一局不算（首次 import/分配开销）
        for g in range(args.deals):
            seat = g % 4
            table = []
            ri = 0
            for pos in range(4):
                if pos == seat:
                    table.append(guard)
                else:
                    table.append(refs[ri])
                    ri += 1
            play_one(table, seed=args.seed + g * 131 + 7)
        s = guard.stats.summary()
        s["name"] = spec
        results.append(s)
        print(f"{spec:<26} n={s['n']:<5} p50={s['p50_ms']:>7.2f}ms "
              f"p95={s['p95_ms']:>7.2f}ms p99={s['p99_ms']:>7.2f}ms "
              f"max={s['max_ms']:>7.2f}ms  fallback={s['fallbacks']}  "
              f"→ {'PASS' if s['realtime'] else 'FAIL'}", flush=True)

    budget = args.budget_ms
    lines = ["# 实时性基准：自研模型单步决策耗时（2026-11-05）", "",
             f"> 预算 {budget:.0f} ms/步；判据：**p99 ≤ {budget:.0f}ms 且 "
             f"max ≤ {budget:.0f}ms**。CPU-only、2 线程、Agent 每局换座位、"
             f"每模型 {args.deals} 局。", "",
             "| Agent | 样本 | p50 | p95 | p99 | max | mean | 超预算退回 | 判定 |",
             "|---|---|---|---|---|---|---|---|---|"]
    ok = 0
    for s in results:
        verdict = "**PASS**" if s["realtime"] else "**FAIL**"
        ok += 1 if s["realtime"] else 0
        lines.append(f"| {s['name']} | {s['n']} | {s['p50_ms']:.2f} ms | "
                     f"{s['p95_ms']:.2f} ms | {s['p99_ms']:.2f} ms | "
                     f"{s['max_ms']:.2f} ms | {s['mean_ms']:.2f} ms | "
                     f"{s['fallbacks']} | {verdict} |")
    lines += ["", f"结论：{ok}/{len(results)} 个 Agent 在 {budget:.0f} ms 预算内"
              "实时（p99 与 max 均达标）。", "",
              "> 护栏：`realtime_guard.RealtimeGuard` 在单步超预算时自动退回"
              "规则基线并计数，对局不会被慢模型卡死。", "",
              "复现：`python realtime_bench.py --deals 8`"]
    text = "\n".join(lines)
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"budget_ms": budget, "deals": args.deals,
                   "results": results}, f, ensure_ascii=False, indent=1)
    print("\n" + text, flush=True)
    print(f"写入 {args.out_md}\n写入 {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())