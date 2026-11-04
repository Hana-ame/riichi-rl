"""听牌率 ↔ 放铳率审计：验证「放铳率低是因为听牌率低」这个假设。

背景（用户 2026-11-05）：「放铳率太低了，听牌率普遍低的关系。」
做法：同一批牌局里，记录每个 seat 是否**本局进入过听牌**（代码判定，规则信息）、
流局时是否听牌、是否有荣和机会、实际放铳/和牌；再按「本局是否有人进过听牌」
分组统计放铳率（条件放铳率 vs 无条件放铳率）。

用法：
    python tenpai_audit.py --deals 80
    python tenpai_audit.py --agents "greedy_tenpai,bc_v1_last,teacher47_v3,hybrid:teacher47_v3"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cpu_only  # noqa: F401
import torch

from arena_elo import RuleAgent, play_one, CKPT
from cpu_only import set_inference_threads
from version_benchmark import make_agent

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT = ("greedy_tenpai,bc_v1_last,teacher47_v3,hybrid:teacher47_v3,"
           "hybridd:teacher47_v3")


def build_agent(spec):
    base = spec.split(":", 1)[1] if spec.startswith(
        ("hybrid:", "hybridd:", "hybridw:")) else spec
    path = os.path.join(CKPT, base if base.endswith(".pt") else base + ".pt")
    if spec in ("greedy_tenpai", "call_first", "yaochu_first", "random"):
        return RuleAgent(spec)
    return make_agent(spec, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", default=DEFAULT)
    ap.add_argument("--deals", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--out-json", default=os.path.join(
        ROOT, "rules", "TENPAI_DEALIN_2026-11-05.json"))
    ap.add_argument("--out-md", default=os.path.join(
        ROOT, "rules", "TENPAI_DEALIN_2026-11-05.md"))
    args = ap.parse_args()

    set_inference_threads(2)
    specs = [x.strip() for x in args.agents.split(",") if x.strip()]
    # 每个目标 agent 固定坐一席，其余三席用 greedy_tenpai（快、会立直）
    refs = [RuleAgent("greedy_tenpai") for _ in range(3)]
    rows = {}
    t0 = time.time()
    for spec in specs:
        agent = build_agent(spec)
        st = {"deals": 0, "reached_tenpai": 0, "tenpai_at_draw": 0,
              "ron_legal": 0, "dealin": 0, "wins": 0, "draws": 0,
              "opp_tenpai_deals": 0, "opp_tenpai_dealin": 0,
              "no_tenpai_deals": 0}
        for g in range(args.deals):
            seat = g % 4
            agents = []
            ri = 0
            for pos in range(4):
                if pos == seat:
                    agents.append(agent)
                else:
                    agents.append(refs[ri])
                    ri += 1
            # 跟踪：目标/三家 本局是否进过听牌
            env_holder = {}
            reached = [False, False, False, False]

            def on_step(env):
                for i in range(4):
                    if not reached[i] and env._is_tenpai(i):
                        reached[i] = True
            # play_one 内部不暴露逐步回调，直接复制一个轻量循环
            from mahjong import env as E
            from mahjong.hanchan import HanchanState
            h = HanchanState(dealer=(args.seed + g * 131) % 4)
            env = h.new_env(seed=args.seed + g * 131)
            for a in agents:
                a.set_game(env, h)          # HanchanAgent/编码需要 hstate
            r = env.start_turn()
            steps = 0
            while r is None and steps < 500:
                steps += 1
                on_step(env)
                acts = {}
                for p in env.pending_players():
                    legal = env.legal_actions(p)
                    if not legal:
                        acts[p] = E.PASS_ACTION
                        continue
                    acts[p] = agents[p].act(env, p, legal)
                r = env.step(acts)
            if r is None:
                continue
            st["deals"] += 1
            target = seat
            if reached[target]:
                st["reached_tenpai"] += 1
            if r.result_type == "ryuukyoku":
                st["draws"] += 1
                if r.tenpai[target]:
                    st["tenpai_at_draw"] += 1
            if r.result_type == "ron" and r.from_player == target:
                st["dealin"] += 1
            if r.result_type != "ryuukyoku" and r.winner == target:
                st["wins"] += 1
            # 对手（非目标）是否进过听牌
            opp_tenpai = any(reached[i] for i in range(4) if i != target)
            if opp_tenpai:
                st["opp_tenpai_deals"] += 1
                if r.result_type == "ron" and r.from_player == target:
                    st["opp_tenpai_dealin"] += 1
            if not any(reached):
                st["no_tenpai_deals"] += 1
        n = max(1, st["deals"])
        st["tenpai_rate"] = round(100.0 * st["reached_tenpai"] / n, 1)
        st["tenpai_at_draw_rate"] = round(
            100.0 * st["tenpai_at_draw"] / max(1, st["draws"]), 1)
        st["win_rate"] = round(100.0 * st["wins"] / n, 1)
        st["deal_in_rate"] = round(100.0 * st["dealin"] / n, 1)
        st["cond_deal_in_rate"] = round(
            100.0 * st["opp_tenpai_dealin"] / max(1, st["opp_tenpai_deals"]), 1)
        st["no_tenpai_rate"] = round(100.0 * st["no_tenpai_deals"] / n, 1)
        rows[spec] = st
        print(f"{spec:<24} 听牌 {st['tenpai_rate']:5.1f}%  "
              f"和牌 {st['win_rate']:5.1f}%  放铳 {st['deal_in_rate']:4.1f}%  "
              f"条件放铳(对手进听时) {st['cond_deal_in_rate']:5.1f}%  "
              f"({st['deals']}局, {time.time()-t0:.0f}s)", flush=True)

    lines = ["# 听牌率 ↔ 放铳率审计（2026-11-05）", "",
             f"> 每个 Agent 坐一席、其余三家 greedy_tenpai，各 {args.deals} 局；",
             "> 听牌=本局任意时刻 `env._is_tenpai` 为真（规则判定）；",
             "> 条件放铳=只在「有对手本局进过听牌」的局里统计放铳率。", "",
             "| Agent | 听牌率 | 流局时听牌率 | 和牌率 | 放铳率 | 条件放铳率 | 本局无人听牌率 |",
             "|---|---|---|---|---|---|---|"]
    for spec, s in rows.items():
        lines.append(
            f"| {spec} | {s['tenpai_rate']}% | {s['tenpai_at_draw_rate']}% | "
            f"{s['win_rate']}% | {s['deal_in_rate']}% | "
            f"{s['cond_deal_in_rate']}% | {s['no_tenpai_rate']}% |")
    lines += ["", "## 判读", "",
              "- 若「未听牌局」占比高、且条件放铳率明显高于无条件放铳率，",
              "  则放铳率低主要是**听牌/和牌机会少**导致的，不是防守好。"]
    text = "\n".join(lines)
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"deals": args.deals, "agents": rows}, f,
                  ensure_ascii=False, indent=1)
    print("\n" + text, flush=True)
    print(f"写入 {args.out_md}\n写入 {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())