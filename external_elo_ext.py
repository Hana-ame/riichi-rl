"""外部模型 session 级 Elo（v2.1，2026-11-27 修复版）

已修复：
1. 场均净得分改为终局 h.sticks[i] - 25000（含立直供托），分母为各选手实际出场场数 sessions_played[n]。
2. 补齐按「每一局」统计的和牌率（wins/hands）与放铳率（dealins/hands），精确读取 r.winner / r.from_player。
3. 修复 reselect 截断 bug：8 人阵容下真正保留前一半中的主力并随机换入落选者组成 4 人桌。
4. 修复同局复牌 seed 冲突：使用 seed + total_hands，避免连庄/南场重复发牌。
5. mahjonglm100 / mahjonglm10 改用仓库内 v2 适配器 external_mahjonglm.py，并在每步后调用 observe(env, acts)。
"""

from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cpu_only  # noqa: F401
from cpu_only import set_inference_threads
from arena_elo import RuleAgent, CKPT
from mahjong import env as E
from mahjong.hanchan import HanchanState
from version_benchmark import make_agent

EXT_SPECS = {"suphx", "riichippo", "mahjonglm100", "mahjonglm10"}
START_POINTS = 25000


def load(spec: str):
    """按 spec 加载 agent。外部模型直接构造；自研走 make_agent。"""
    if spec in ("greedy_tenpai", "call_first", "random", "damaten"):
        return RuleAgent(spec)
    if spec == "suphx":
        from external_suphx import SuphxAgent
        return SuphxAgent("suphx")
    if spec == "riichippo":
        from external_riichippo import RiichippoAgent
        return RiichippoAgent("riichippo", ckpt="external/riichippo/v1.pth")
    if spec == "mahjonglm100":
        from external_mahjonglm import MahjongLMAgent
        return MahjongLMAgent("external/mahjonglm100", seat_name="mahjonglm100")
    if spec == "mahjonglm10":
        from external_mahjonglm import MahjongLMAgent
        return MahjongLMAgent("external/mahjonglm10", seat_name="mahjonglm10")
    return make_agent(spec, os.path.join(CKPT, spec + ".pt"))


def elo_update_4p(r, placement, K=24):
    """placement = [4 个 spec 名按最终排名]，逐对零和更新。"""
    for i in range(4):
        for j in range(i + 1, 4):
            a, b = placement[i], placement[j]
            sa = 1.0 if i < j else 0.0
            ea = 1.0 / (1.0 + 10 ** ((r[b] - r[a]) / 400.0))
            r[a] += K * (sa - ea)
            r[b] -= K * (sa - ea)


class SessionStats:
    """累计每名选手的出场场数、总局数（每座位每局）、和牌次数、放铳次数与净点数。"""

    def __init__(self):
        self.sessions_played = {}
        self.hands = {}
        self.wins = {}
        self.dealins = {}
        self.pts = {}
        self.top1 = {}

    def track(self, spec: str):
        if spec not in self.hands:
            self.sessions_played[spec] = 0
            self.hands[spec] = 0
            self.wins[spec] = 0
            self.dealins[spec] = 0
            self.pts[spec] = 0.0
            self.top1[spec] = 0

    def record_session(self, names, placement, net_pts, hand_wins, hand_dealins, total_hands):
        for i, n in enumerate(names):
            self.track(n)
            self.sessions_played[n] += 1
            self.hands[n] += total_hands
            self.pts[n] += net_pts[i]
            self.wins[n] += hand_wins[i]
            self.dealins[n] += hand_dealins[i]
        if placement:
            self.track(placement[0])
            self.top1[placement[0]] += 1


def run_half_game(names, agent_pool, seed, east_only=False):
    """打完整半庄/东风 session。

    返回:
        placement: [4 个 spec 名按最终点数排名]
        net_pts: [4 个座位的终局净点数 h.sticks[i] - 25000]
        hand_wins: [4 个座位在本场内的和牌局数]
        hand_dealins: [4 个座位在本场内的放铳局数]
        total_hands: 本场总局数（kyoku 次数）
    """
    h = HanchanState(dealer=seed % 4, start_points=START_POINTS, east_only=east_only)
    seats_agents = [agent_pool[n] for n in names]

    hand_wins = [0, 0, 0, 0]
    hand_dealins = [0, 0, 0, 0]
    total_hands = 0

    while True:
        env = h.new_env(seed=seed + total_hands)
        for a in seats_agents:
            if hasattr(a, "set_game"):
                a.set_game(env, h)

        r = env.start_turn()
        steps = 0
        while r is None and steps < 600:
            steps += 1
            acts = {}
            for p in env.pending_players():
                legal = env.legal_actions(p)
                acts[p] = seats_agents[p].act(env, p, legal) if legal else E.PASS_ACTION
            r = env.step(acts)
            for a in seats_agents:
                if hasattr(a, "observe"):
                    a.observe(env, acts)

        if r is None:
            break

        total_hands += 1
        if r.result_type != "ryuukyoku" and r.winner is not None:
            hand_wins[r.winner] += 1
            if r.result_type == "ron" and r.from_player is not None:
                hand_dealins[r.from_player] += 1

        h.advance(r)
        if h.finished:
            break

    placement = [names[i] for i in h.final_placement()]
    net_pts = [float(h.sticks[i] - START_POINTS) for i in range(4)]
    return placement, net_pts, hand_wins, hand_dealins, total_hands


def reselect(ratings, pool_names, rng):
    """审阅口径：Elo 前一半保留主力 + 从后一半随机替换组成 4 人桌。"""
    if len(pool_names) <= 4:
        return list(pool_names)
    sorted_names = sorted(pool_names, key=lambda n: -ratings.get(n, 1500.0))
    half = max(2, len(pool_names) // 2)
    top_half = sorted_names[:half]
    bottom_half = sorted_names[half:]
    # 每桌 4 人：从前一半抽 2-3 人 + 从后一半随机替换 1-2 人，避免截断丢弃随机补位
    n_fill = min(max(1, min(2, len(bottom_half))), 2)
    n_keep = 4 - n_fill
    keep = rng.sample(top_half, min(n_keep, len(top_half)))
    fill = rng.sample(bottom_half, min(n_fill, len(bottom_half)))
    chosen = list(dict.fromkeys(keep + fill))
    for n in sorted_names:
        if len(chosen) >= 4:
            break
        if n not in chosen:
            chosen.append(n)
    return chosen[:4]


def run_config(mode: str, roster: list[str], sessions: int, seed: int,
               reselect_enabled: bool = False, rng: random.Random | None = None):
    """跑一个模式（east 或 half），N 场，返回按场与按局完整统计。"""
    east_only = (mode == "east")
    rating = {n: 1500.0 for n in roster}
    stats = SessionStats()
    for n in roster:
        stats.track(n)
    pool_agents = {n: load(n) for n in roster}

    active = roster[:4] if len(roster) >= 4 else list(roster)
    hands_per_session = []
    for s in range(sessions):
        if reselect_enabled and rng is not None and (s == 0 or s % max(1, sessions // 4) == 0):
            active = reselect(rating, roster, rng)
        shift = s % 4
        names = active[shift:] + active[:shift]
        game_seed = seed + s * 977
        placement, net_pts, hand_wins, hand_dealins, total_hands = run_half_game(
            names, pool_agents, game_seed, east_only=east_only
        )
        elo_update_4p(rating, placement)
        stats.record_session(names, placement, net_pts, hand_wins, hand_dealins, total_hands)
        hands_per_session.append(total_hands)
        print(f"[{mode}] {s+1}/{sessions}  hands={total_hands}  active={active}", flush=True)

    per_spec = {}
    for n in roster:
        sp = stats.sessions_played[n]
        hd = stats.hands[n]
        per_spec[n] = {
            "elo": round(rating[n], 1),
            "sessions_played": sp,
            "hands_played": hd,
            "avg_pts_per_session": round(stats.pts[n] / sp, 1) if sp else 0.0,
            "win_rate_per_hand": round(stats.wins[n] / hd, 4) if hd else 0.0,
            "dealin_rate_per_hand": round(stats.dealins[n] / hd, 4) if hd else 0.0,
            "wins": stats.wins[n],
            "dealins": stats.dealins[n],
            "top1": stats.top1[n],
        }

    return {
        "mode": mode,
        "sessions": sessions,
        "avg_hands_per_session": round(sum(hands_per_session) / max(1, len(hands_per_session)), 2),
        "elo": {n: per_spec[n]["elo"] for n in roster},
        "avg_pts_per_session": {n: per_spec[n]["avg_pts_per_session"] for n in roster},
        "per_spec": per_spec,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=8, help="每种模式跑几场")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--roster", default="teacher47_eff_wuke_d2,suphx,riichippo,mahjonglm100,greedy_tenpai")
    ap.add_argument("--modes", default="half,east", help="逗号分隔：half,east")
    ap.add_argument("--reselect", action="store_true", help="启用前一半+随机替换")
    ap.add_argument("--out-md", default="rules/EXTERNAL_ELO_EXT_2026-11-28.md")
    ap.add_argument("--out-json", default="rules/EXTERNAL_ELO_EXT_2026-11-28.json")
    args = ap.parse_args()

    set_inference_threads(2)
    roster = [s.strip() for s in args.roster.split(",") if s.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    rng = random.Random(args.seed)

    all_results = {}
    t0 = time.time()
    for mode in modes:
        print(f"=== 模式 {mode} 开始 ===", flush=True)
        res = run_config(mode, roster, args.sessions, args.seed, args.reselect, rng)
        all_results[mode] = res
        print(f"=== 模式 {mode} 完成，用时 {time.time()-t0:.0f}s ===", flush=True)

    lines = [
        "# 外部模型 session 级 Elo（修复版）",
        "",
        f"> 阵容：{', '.join(roster)}",
        f"> 每模式 {args.sessions} 场，东风+半庄双报；"
        + ("启用前一半+随机替换组队；" if args.reselect else "")
        + "座位每场轮换；场均净点 = 终局点数 - 25000（按实际出场场数平均）；和牌率/放铳率按每座位每局统计。",
        "> 注：外部模型仅接管其适配器支持的决策（非同构对比，结果仅供横向参考）。",
        "",
    ]
    for mode in modes:
        res = all_results[mode]
        mode_zh = "半庄" if mode == "half" else "东风"
        lines += [
            f"## {mode_zh}（{res['sessions']} 场，场均 {res['avg_hands_per_session']:.1f} 局）",
            "",
            "| 参赛者 | 类型 | 出场场数 | 总局数 | Elo | 场均净点 | 和牌率/局 | 放铳率/局 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        ps = res["per_spec"]
        for n in sorted(ps, key=lambda x: -ps[x]["elo"]):
            typ = "外部" if n in EXT_SPECS else "自研"
            row = ps[n]
            lines.append(
                f"| {n} | {typ} | {row['sessions_played']} | {row['hands_played']} "
                f"| **{row['elo']:.1f}** | {row['avg_pts_per_session']:+.1f} "
                f"| {row['win_rate_per_hand']*100:.1f}% | {row['dealin_rate_per_hand']*100:.1f}% |"
            )
        lines.append("")

    md_text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md_text)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(md_text)
    print(f"总用时 {time.time()-t0:.0f}s，写入 {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
