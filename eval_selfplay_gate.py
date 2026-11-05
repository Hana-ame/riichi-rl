"""4 席自对弈硬门槛评测：每局每座位分母的和牌率/放铳率（带 Wilson 95% CI）。

用户硬门槛（2026-11-05）：
  - 每局每座位和牌率 15–25%
  - 每局每座位放铳率 9–20%
旧口径问题：`eval_hanchan` 只报「局和牌」（分母是局），`tenpai_audit` 是
「单席 vs 3 greedy」；都不等于「4 席同版本自对弈」的门槛口径。本脚本补这条。

用法：
  python eval_selfplay_gate.py --ckpt checkpoints/teacher47_v3.pt --deals 60
  python eval_selfplay_gate.py --policy greedy_tenpai --deals 60
"""
from __future__ import annotations
import argparse, os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpu_only  # noqa: F401
import torch
torch.set_num_threads(2)   # 必须在 cpu_only 之后覆盖：它把线程数设成全核，
                           # 多进程/评测并行时会互相抢核反而挂住（实测 1082% CPU 空转）
from mahjong import env as E
from mahjong import refengine as REF
from mahjong import policies as POL
from mahjong.hanchan import HanchanState
from version_benchmark import make_agent


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (100 * (c - m) / d, 100 * (c + m) / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--policy", default="", help="规则策略名（与 --ckpt 二选一）")
    ap.add_argument("--deals", type=int, default=60)
    ap.add_argument("--seed0", type=int, default=91000)
    a = ap.parse_args()
    if a.ckpt:
        path = a.ckpt if a.ckpt.endswith(".pt") else os.path.join("checkpoints", a.ckpt + ".pt")
        label = os.path.basename(path).replace(".pt", "")
        agents = [make_agent(label, path) for _ in range(4)]
    elif a.policy:
        label = a.policy
        agents = [POL.make_policy(a.policy, seed=i) for i in range(4)]
    else:
        raise SystemExit("需要 --ckpt 或 --policy")
    seat_w = [0] * 4; seat_d = [0] * 4; ryu = 0; ten = 0; nten = 0
    t0 = time.time()
    for g in range(a.deals):
        h = HanchanState(dealer=g % 4); env = h.new_env(seed=a.seed0 + g)
        ags = ([make_agent(label, path) for _ in range(4)] if a.ckpt else
               [POL.make_policy(a.policy, seed=g * 97 + i) for i in range(4)])
        for ag in ags:
            if hasattr(ag, "set_game"):
                ag.set_game(env, h)
        r = env.start_turn(); steps = 0
        while r is None and steps < 500:
            steps += 1; acts = {}
            for p in env.pending_players():
                legal = env.legal_actions(p)
                if not legal:
                    acts[p] = E.PASS_ACTION; continue
                ag = ags[p]
                act = ag.choose(env, p, legal) if isinstance(ag, POL.Policy) else ag.act(env, p, legal)
                if act not in legal:
                    act = E.PASS_ACTION if E.PASS_ACTION in legal else legal[0]
                acts[p] = act
            r = env.step(acts)
        k = r.result_type if r is not None else "HANG"
        if k == "ryuukyoku":
            ryu += 1
        elif k not in ("HANG",):
            w = r.winner
            seat_w[w] += 1
            if k == "ron" and r.from_player is not None:
                seat_d[r.from_player] += 1
        for p in range(4):
            nten += 1
            if REF.ref_shanten(env.players[p].full_counts()) == 0 or (
                    r is not None and k not in ("ryuukyoku", "HANG") and getattr(r, "winner", None) == p):
                ten += 1
    N = 4 * a.deals
    W = sum(seat_w); D = sum(seat_d)
    wr = 100 * W / N; dr = 100 * D / N
    wci = wilson(W, N); dci = wilson(D, N)
    print(f"=== 4 席自对弈硬门槛：{label}  {a.deals} 局 ({N} 座位·局) {time.time()-t0:.0f}s ===")
    print(f"  每座位和牌 {W}/{N} = {wr:.2f}%  CI[{wci[0]:.1f},{wci[1]:.1f}]  "
          f"门槛 15–25% -> {'PASS' if 15 <= wr <= 25 else 'FAIL'}")
    print(f"  每座位放铳 {D}/{N} = {dr:.2f}%  CI[{dci[0]:.1f},{dci[1]:.1f}]  "
          f"门槛 9–20%  -> {'PASS' if 9 <= dr <= 20 else 'FAIL'}")
    print(f"  流局 {ryu}/{a.deals} = {100*ryu/a.deals:.1f}%  终局听牌 {100*ten/nten:.1f}%")
    print(f"  分座位和牌 {seat_w}  分座位放铳 {seat_d}")
    out = {"label": label, "deals": a.deals, "seat_deals": N,
           "seat_wins": W, "seat_win_rate": wr, "seat_win_ci": wci,
           "seat_dealins": D, "seat_deal_in_rate": dr, "seat_deal_in_ci": dci,
           "win_pass": 15 <= wr <= 25, "deal_in_pass": 9 <= dr <= 20,
           "ryuukyoku": ryu, "tenpai_at_end": 100 * ten / nten,
           "per_seat_wins": seat_w, "per_seat_dealins": seat_d}
    p = f"rules/SELFPLAY_GATE_{label}.json"
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=1)
    print(f"  写入 {p}")


if __name__ == "__main__":
    main()
