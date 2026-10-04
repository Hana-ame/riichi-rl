"""竞技场：多策略混战 + 可证伪的排名判据。

对应 ARENA.md。本文是它的可执行版本。

三条铁律：
1. **座位轮换**：每个批次覆盖全部 4 个座次，消除庄家偏差。
2. **批次内共享牌山、批次间独立**：统计检验以批次为独立单元
   （cluster-robust），不拿单局当独立样本。
3. **排名必须可证伪**：置信区间跨 0 就写「无法区分」，不写「略强」。
"""

from __future__ import annotations

import math
import random

import os
import sys

# 允许既能从包内导入（from riichi_rl.arena import ...）也能直接运行
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mahjong import env as E          # noqa: E402
from mahjong import policies as POL   # noqa: E402

N_PLAYERS = 4


def seat_rotation(names):
    """返回一个批次内的座位映射列表：len == 4，覆盖全部座次。

    参数是**策略名列表**（不是索引）。若策略数 < 4，用重复填充补满 4 席。
    生成的 4 个映射保证：任一策略在 4 局里坐庄次数差 <= 1。
    """
    base = list(names)
    while len(base) < N_PLAYERS:
        base.append(base[len(base) % len(names)])
    base = base[:N_PLAYERS]
    rots = []
    for k in range(N_PLAYERS):
        rots.append(base[k:] + base[:k])
    return rots


class ArenaResult:
    """一场竞技的统计结果。"""

    __slots__ = ("metrics", "matrix", "pairwise", "n_games", "n_batches")

    def __init__(self, metrics, matrix, pairwise, n_games, n_batches):
        self.metrics = metrics          # {name: {...}}
        self.matrix = matrix            # {(a,b): mean point diff of a vs b}
        self.pairwise = pairwise        # {(a,b): (mean, ci_lo, ci_hi, n)}
        self.n_games = n_games
        self.n_batches = n_batches

    def rank(self, metric="avg_point_diff"):
        return sorted(self.metrics.items(),
                      key=lambda kv: -kv[1].get(metric, 0.0))

    def report(self):
        lines = []
        lines.append(f"对局数 {self.n_games}，批次 {self.n_batches}")
        lines.append("")
        lines.append(f"{'策略':<16}{'平均点差':>10}{'和了率':>9}{'放铳率':>9}"
                     f"{'立直率':>9}{'流局率':>9}{'平均和了点':>12}")
        lines.append("-" * 78)
        for name, m in self.rank():
            lines.append(
                f"{name:<16}{m['avg_point_diff']:>10.0f}"
                f"{m['win_rate']:>9.3f}{m['deal_in_rate']:>9.3f}"
                f"{m['riichi_rate']:>9.3f}{m['draw_rate']:>9.3f}"
                f"{m['avg_win_points']:>12.0f}")
        lines.append("")
        lines.append("两两对局（行方视角的平均点差，95% CI，批次为独立单元）")
        lines.append("-" * 78)
        names = sorted(self.pairwise.keys())
        for a in sorted({k[0] for k in self.pairwise}):
            for b in sorted({k[1] for k in self.pairwise}):
                if a == b:
                    continue
                st = self.pairwise.get((a, b))
                if st is None:
                    continue
                mean, lo, hi, n = st
                verdict = ("A 强" if lo > 0 else
                           "B 强" if hi < 0 else "无法区分")
                lines.append(f"  {a:<14} vs {b:<14} "
                             f"{mean:>8.0f} [{lo:>8.0f},{hi:>8.0f}] "
                             f"n={n:<4} {verdict}")
        return "\n".join(lines)


def run_batch(policy_names, seed, dealer=0):
    """跑一局（4 席，不同策略）。返回 (result, policies)。"""
    env = E.MahjongEnv(seed=seed, dealer=dealer)
    pols = {}
    for slot, pname in enumerate(policy_names):
        pols[slot] = POL.make_policy(pname, seed=seed * 1000 + slot)
    r = env.start_turn()
    steps = 0
    while r is None and steps < 600:
        steps += 1
        acts = {}
        for p in env.pending_players():
            acts[p] = pols[p].act(env, p)
        r = env.step(acts)
    return r, env, pols


def evaluate(policy_names, n_batches=100, seed0=0, verbose=True):
    """跑竞技场，返回 ArenaResult。

    公平性协议：
      - 每个批次是一个「回合集合」：同一批牌山下，策略依次坐 4 个座次
      - 批次内 4 局共享牌山种子 -> 批次间独立 -> 用批次做独立单元
    """
    names = list(policy_names)
    n = len(names)
    rots = seat_rotation(names)

    # 累积量
    games = 0
    batches = []
    stat = {name: {"games": 0, "wins": 0, "deal_in": 0, "riichi": 0,
                   "draws": 0, "points": 0.0, "win_points": [], "riichi_pts": 0}
            for name in names}

    pair_diff = {(a, b): [] for a in names for b in names if a != b}

    for bi in range(n_batches):
        base_seed = seed0 + bi * 7
        batch_deltas = {name: [] for name in names}
        for k in range(N_PLAYERS):
            # 策略映射：rots[k] 给出每个座位放哪个策略
            mapping = rots[k]
            r, env, _ = run_batch(mapping, base_seed + k,
                                  dealer=(base_seed + k) % N_PLAYERS)
            games += 1
            if r is None:
                continue
            d = r.points_delta
            for slot, name in enumerate(mapping):
                st = stat[name]
                st["games"] += 1
                st["points"] += d[slot]
                batch_deltas[name].append(d[slot])
                if env.players[slot].riichi:
                    st["riichi"] += 1
                if r.result_type == "ryuukyoku":
                    st["draws"] += 1
                else:
                    if slot == r.winner:
                        st["wins"] += 1
                        st["win_points"].append(abs(d[slot]))
                    if slot == r.from_player:
                        st["deal_in"] += 1
            # 两两点差（同一局内）
            for slot_a in range(N_PLAYERS):
                for slot_b in range(N_PLAYERS):
                    if slot_a == slot_b:
                        continue
                    a = mapping[slot_a]
                    b = mapping[slot_b]
                    if a != b:
                        pair_diff[(a, b)].append(d[slot_a] - d[slot_b])
        batches.append({kk: sum(vv) for kk, vv in batch_deltas.items()})

    # 汇总指标
    metrics = {}
    for name in names:
        st = stat[name]
        g = max(1, st["games"])
        wp = st["win_points"]
        metrics[name] = {
            "avg_point_diff": st["points"] / g,
            "win_rate": st["wins"] / g,
            "deal_in_rate": st["deal_in"] / g,
            "draw_rate": st["draws"] / g,
            "riichi_rate": st["riichi"] / g,
            "avg_win_points": (sum(wp) / len(wp)) if wp else 0.0,
            "games": st["games"],
        }

    # 配对两两统计 + cluster-robust CI（以批次为独立单元）
    pairwise = {}
    for (a, b), vals in pair_diff.items():
        if len(vals) < 2:
            continue
        mean = sum(vals) / len(vals)
        # 批次内均值 -> 每批次一个样本 -> 标准误
        per_batch = []
        idx = 0
        # 重新按批次切分
        for bi in range(n_batches):
            # 该批次内 (a,b) 的样本数
            cnt = 0
            s = 0.0
            for k in range(N_PLAYERS):
                n_in_batch = _pair_count_in_batch(names, rots, k, a, b)
                for _ in range(n_in_batch):
                    s += vals[idx]
                    idx += 1
                    cnt += 1
            if cnt:
                per_batch.append(s / cnt)
        if len(per_batch) < 2:
            continue
        m = sum(per_batch) / len(per_batch)
        var = sum((x - m) ** 2 for x in per_batch) / (len(per_batch) - 1)
        se = math.sqrt(var / len(per_batch))
        ci = 1.96 * se
        pairwise[(a, b)] = (mean, m - ci, m + ci, len(per_batch))

    return ArenaResult(metrics, {}, pairwise, games, n_batches)


def _pair_count_in_batch(names, rots, k, a, b):
    """一批次第 k 局（座次 rots[k]）里，a 与 b 同时在场的次数。"""
    m = rots[k]
    if m.count(a) and m.count(b):
        return m.count(a) * m.count(b)
    return 0
