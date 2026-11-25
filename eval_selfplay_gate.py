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
    ap.add_argument("--ensemble", default="", help="逗号分隔的多个 ckpt = 集成（logits 平均）")
    ap.add_argument("--danger-ckpt", default="", help="模型 + 危险分惩罚（λ 见 --danger-lam）")
    ap.add_argument("--danger-lam", type=float, default=5.0)
    ap.add_argument("--danger-riichi-only", action="store_true",
                    help="只在有对手立直时才施加危险惩罚")
    ap.add_argument("--call-head", default="", help="副露二分类头 ckpt（模型决定叫/不叫）")
    ap.add_argument("--call-thr", type=float, default=0.5)
    ap.add_argument("--post-call", action="store_true",
                    help="启用方案 1A：Post-Call State Value 叫后前瞻副露决策器")
    ap.add_argument("--post-call-margin", type=float, default=0.02,
                    help="Post-Call V(call)-V(pass) 最小正收益阈值")
    ap.add_argument("--dora-ckpt", default="", help="宝牌策略模型（aux 头驱动立牌价值）")
    ap.add_argument("--dora-lam", type=float, default=1.0)
    ap.add_argument("--model-choice-ckpt", default="",
                    help="模型自己决定立直（去掉规则强制立直）")
    ap.add_argument("--tenhou-ckpt", default="",
                    help="Tenhou 离线 MLP（tenhou_agent.TenhouMLPAgent）")
    ap.add_argument("--dual-stream-ckpt", default="",
                    help="DualStreamMahjongNet（dual_stream_agent.DualStreamAgent）")
    ap.add_argument("--hybrid-call", default="",
                    help="HybridCallAgent：叫牌用此 ckpt，弃牌用 --ckpt")
    ap.add_argument("--mode", default="hanchan", choices=["hanchan", "east"],
                    help="hanchan=半庄 E1-S4，east=东风战 E1-E4")
    ap.add_argument("--dora-keep", action="store_true", help="宝牌保值：压低打宝牌的 logit")
    ap.add_argument("--deals", type=int, default=60)
    ap.add_argument("--seed0", type=int, default=91000)
    a = ap.parse_args()
    ens_paths = []
    if a.ensemble:
        ens_paths = [os.path.join("checkpoints", n if n.endswith(".pt") else n + ".pt")
                     for n in a.ensemble.split(",")]
    if a.tenhou_ckpt:
        path = a.tenhou_ckpt if a.tenhou_ckpt.endswith(".pt") else os.path.join("checkpoints", a.tenhou_ckpt + ".pt")
        label = f"{os.path.basename(path).replace('.pt','')}"
    elif a.model_choice_ckpt:
        path = os.path.join("checkpoints", a.model_choice_ckpt if a.model_choice_ckpt.endswith(".pt")
                            else a.model_choice_ckpt + ".pt")
        label = f"{os.path.basename(path).replace('.pt','')}_modelchoice"
    elif a.dora_ckpt:
        if a.dora_ckpt == "self":     # 用 --ckpt 当主模型，只加宝牌价值（无 aux 头）
            path = os.path.join("checkpoints", a.ckpt if a.ckpt.endswith(".pt")
                                else a.ckpt + ".pt")
        else:
            path = os.path.join("checkpoints", a.dora_ckpt if a.dora_ckpt.endswith(".pt")
                                else a.dora_ckpt + ".pt")
        label = f"{os.path.basename(path).replace('.pt','')}_dora{a.dora_lam}" + (
            "_keep" if a.dora_keep else "")
    elif a.call_head:
        path = os.path.join("checkpoints", a.ckpt if a.ckpt.endswith(".pt")
                            else a.ckpt + ".pt")
        label = f"{os.path.basename(path).replace('.pt','')}_callhead{a.call_thr}"
    elif a.danger_ckpt:
        path = os.path.join("checkpoints", a.danger_ckpt if a.danger_ckpt.endswith(".pt")
                            else a.danger_ckpt + ".pt")
        label = f"{os.path.basename(path).replace('.pt','')}_danger{int(a.danger_lam)}" + (
            "_riichionly" if a.danger_riichi_only else "")
    elif a.ensemble:
        from ensemble_agent import EnsembleAgent
        label = "ens_" + "+".join(os.path.basename(p).replace(".pt", "") for p in ens_paths)
    elif a.ckpt:
        path = a.ckpt if a.ckpt.endswith(".pt") else os.path.join("checkpoints", a.ckpt + ".pt")
        label = os.path.basename(path).replace(".pt", "") + ("_postcall" if a.post_call else "")
    elif a.policy:
        label = a.policy
        agents = [POL.make_policy(a.policy, seed=i) for i in range(4)]
    else:
        raise SystemExit("需要 --ckpt 或 --policy")
    seat_w = [0] * 4; seat_d = [0] * 4; ryu = 0; ten = 0; nten = 0
    act_cnt = {"discard": 0, "riichi": 0, "chi": 0, "pon": 0, "kan": 0,
               "tsumo": 0, "ron": 0, "pass": 0}
    if a.deals < 400:
        print(f"[warn] deals={a.deals} < 400：小样本下 Wilson 95% CI 较宽，建议 400-600 局评测硬门槛", flush=True)
    t0 = time.time()
    for g in range(a.deals):
        h = HanchanState(dealer=g % 4, east_only=(a.mode == "east"))
        env = h.new_env(seed=a.seed0 + g)
        if a.dual_stream_ckpt:
            from dual_stream_agent import DualStreamAgent
            ds_p = a.dual_stream_ckpt if a.dual_stream_ckpt.endswith(".pt") else os.path.join("checkpoints", a.dual_stream_ckpt + ".pt")
            label = f"{os.path.basename(ds_p).replace('.pt','')}"
            ags = [DualStreamAgent(label, ds_p) for _ in range(4)]
        elif a.hybrid_call:
            from hybrid_call_agent import HybridCallAgent
            call_p = a.hybrid_call if a.hybrid_call.endswith(".pt") else os.path.join("checkpoints", a.hybrid_call + ".pt")
            disc_p = a.ckpt if a.ckpt.endswith(".pt") else os.path.join("checkpoints", a.ckpt + ".pt")
            label = f"hybrid_call_{os.path.basename(call_p).replace('.pt','')}"
            ags = [HybridCallAgent(label, call_p, disc_p) for _ in range(4)]
        elif a.tenhou_ckpt:
            from model_choice_agent import ModelChoiceAgent
            ags = [ModelChoiceAgent(label, path) for _ in range(4)]
        elif a.dora_ckpt:
            from dora_agent import DoraValueAgent
            ags = [DoraValueAgent(label, path, lam=a.dora_lam, keep=a.dora_keep)
                   for _ in range(4)]
        elif a.post_call:
            from post_call_agent import PostCallValueAgent
            ags = [PostCallValueAgent(path, margin=a.post_call_margin) for _ in range(4)]
        elif a.call_head:
            from call_head import CallHeadAgent
            ags = [CallHeadAgent(label, path, a.call_head, thr=a.call_thr)
                   for _ in range(4)]
        elif a.danger_ckpt:
            from danger_model_agent import ModelDangerAgent
            ags = [ModelDangerAgent(label, path, lam=a.danger_lam,
                                    riichi_only=a.danger_riichi_only) for _ in range(4)]
        elif a.ensemble:
            ags = [EnsembleAgent(label, ens_paths) for _ in range(4)]
        elif a.ckpt:
            ags = [make_agent(label, path) for _ in range(4)]
        else:
            ags = [POL.make_policy(a.policy, seed=g * 97 + i) for i in range(4)]
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
                _t = act.type
                if _t == E.ACT_DISCARD: act_cnt["discard"] += 1
                elif _t == E.ACT_RIICHI: act_cnt["riichi"] += 1
                elif _t == E.ACT_CHI: act_cnt["chi"] += 1
                elif _t == E.ACT_PON: act_cnt["pon"] += 1
                elif _t in (E.ACT_ANKAN, E.ACT_MINKAN, E.ACT_ADD_KAN): act_cnt["kan"] += 1
                elif _t == E.ACT_TSUMO: act_cnt["tsumo"] += 1
                elif _t == E.ACT_RON: act_cnt["ron"] += 1
                elif _t == E.ACT_PASS: act_cnt["pass"] += 1
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
    print(f"  动作计数 {act_cnt}")
    out = {"label": label, "deals": a.deals, "seat_deals": N,
           "seat_wins": W, "seat_win_rate": wr, "seat_win_ci": wci,
           "seat_dealins": D, "seat_deal_in_rate": dr, "seat_deal_in_ci": dci,
           "win_pass": 15 <= wr <= 25, "deal_in_pass": 9 <= dr <= 20,
           "ryuukyoku": ryu, "tenpai_at_end": 100 * ten / nten,
           "per_seat_wins": seat_w, "per_seat_dealins": seat_d,
           "action_counts": act_cnt}
    p = f"rules/SELFPLAY_GATE_{label}.json"
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=1)
    print(f"  写入 {p}")


if __name__ == "__main__":
    main()
