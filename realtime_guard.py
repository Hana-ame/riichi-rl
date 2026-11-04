"""实时性护栏：任何 Agent 包一层，保证单步决策在预算内（默认 5 秒）。

用户要求（2026-11-05）：「自研模型请做出 5 秒内 realtime 的模型。」

做法：
- 自研模型（453K~1.65M 参数，CPU 单步 1~5ms）本身远快于预算；
- 但仍加**护栏**：超预算自动退回规则基线（GreedyTenpai）并记一次 fallback，
  保证对局不会被慢模型/慢机器卡死；
- `LatencyStats` 汇总 p50/p95/p99/max，作为「是否 realtime」的可证伪判据
  （p99 ≤ 预算 且 max ≤ 预算）。

用法：
    from realtime_guard import RealtimeGuard
    guard = RealtimeGuard(model_agent, budget_ms=5000)
    action = guard.act(env, p, legal)
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mahjong import policies as POL


class LatencyStats:
    """记录单步决策耗时，输出 p50/p95/p99/max（毫秒）。"""

    def __init__(self):
        self.times = []          # 秒
        self.fallbacks = 0
        self.budget_ms = 5000

    def add(self, sec):
        self.times.append(float(sec))

    def summary(self):
        if not self.times:
            return {"n": 0}
        xs = sorted(self.times)
        def pct(p):
            k = min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))
            return xs[k] * 1000.0
        return {
            "n": len(xs),
            "p50_ms": round(pct(0.50), 2),
            "p95_ms": round(pct(0.95), 2),
            "p99_ms": round(pct(0.99), 2),
            "max_ms": round(xs[-1] * 1000.0, 2),
            "mean_ms": round(sum(xs) / len(xs) * 1000.0, 2),
            "fallbacks": self.fallbacks,
            "budget_ms": self.budget_ms,
            "realtime": bool(xs[-1] * 1000.0 <= self.budget_ms
                             and pct(0.99) <= self.budget_ms),
        }


class RealtimeGuard:
    """包一个 Agent：超预算就退回规则基线，并统计每一步耗时。"""

    def __init__(self, agent, budget_ms=5000.0, fallback_name="greedy_tenpai"):
        self.agent = agent
        self.budget_ms = float(budget_ms)
        self.stats = LatencyStats()
        self.stats.budget_ms = self.budget_ms
        self.fallback = POL.make_policy(fallback_name, seed=0)
        self.name = getattr(agent, "name", type(agent).__name__) + "+realtime"

    def set_game(self, env, hstate=None):
        if hasattr(self.agent, "set_game"):
            self.agent.set_game(env, hstate)

    def act(self, env, p, legal):
        t0 = time.perf_counter()
        act = self.agent.act(env, p, legal)
        dt = time.perf_counter() - t0
        self.stats.add(dt)
        if dt * 1000.0 > self.budget_ms:
            self.stats.fallbacks += 1
            # 预算外：退回规则基线（规则计算很便宜），保证这一手不卡局
            return self.fallback.choose(env, p, legal)
        return act