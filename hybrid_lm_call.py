"""副露模型化实验：弃牌用规则牌效 eff_uke，**吃/碰/明杠只由外部 LM 决策**。

为什么：审阅要求「吃碰杠 + dora 需有模型策略，不能规则接管」。前面自研的规则
call-teacher 反而掉和牌（12.5%→10.0%），说明「规则叫牌」不是好策略；
本脚本把叫牌决策交给 mahjonglm（模型 logits），观察：
  1) 叫牌次数/种类是否接近外部模型自对弈水平；
  2) 4 席同池自对弈的和牌/放铳是否还能守住门槛。
若有效，可再用该混合教师采数据 → 蒸馏到自研模型（叫牌即模型策略）。

用法：
  python hybrid_lm_call.py --model external/mahjonglm10 --deals 120 --seed0 92000
"""
from __future__ import annotations
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpu_only  # noqa: F401
import torch
torch.set_num_threads(2)
from mahjong import env as E
from mahjong import policies as POL
from mahjong.hanchan import HanchanState
from external_mahjonglm import MahjongLMAgent


class HybridCall:
    """弃牌=eff_uke，叫牌=LM 模型决策，和牌=必和。"""
    def __init__(self, lm):
        self.lm = lm
        self.rule = POL.make_policy("eff_uke", seed=0)
        self.calls = {"chi": 0, "pon": 0, "minkan": 0}

    def set_game(self, env, h):
        self.lm.set_game(env, h)

    def observe(self, env, acts):
        self.lm.observe(env, acts)

    def choose(self, env, p, legal):
        types = {a.type for a in legal}
        for wt in (E.ACT_TSUMO, E.ACT_RON):
            if wt in types:
                return next(a for a in legal if a.type == wt)
        if E.Phase.RESPONSE == env.phase:
            call = [a for a in legal if a.type in (E.ACT_CHI, E.ACT_PON, E.ACT_MINKAN)]
            if call:
                a = self.lm.act(env, p, legal)
                if a.type in (E.ACT_CHI, E.ACT_PON, E.ACT_MINKAN):
                    self.calls[{E.ACT_CHI: "chi", E.ACT_PON: "pon",
                                E.ACT_MINKAN: "minkan"}[a.type]] += 1
                    return a
            # LM 过：由规则补 pass（避免 LM 对无叫牌机会的 pass 逻辑干扰弃牌）
        return self.rule.choose(env, p, legal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="external/mahjonglm10")
    ap.add_argument("--deals", type=int, default=60)
    ap.add_argument("--seed0", type=int, default=92000)
    a = ap.parse_args()
    wins = dealin = ryu = 0
    calls = {"chi": 0, "pon": 0, "minkan": 0}
    t0 = time.time()
    for g in range(a.deals):
        h = HanchanState(dealer=g % 4); env = h.new_env(seed=a.seed0 + g)
        ags = [HybridCall(MahjongLMAgent(a.model, seat_name=f"h{i}")) for i in range(4)]
        for ag in ags:
            ag.set_game(env, h)
        r = env.start_turn(); steps = 0
        while r is None and steps < 500:
            steps += 1; acts = {}
            for p in env.pending_players():
                legal = env.legal_actions(p)
                acts[p] = ags[p].choose(env, p, legal) if legal else E.PASS_ACTION
            r = env.step(acts)
            for ag in ags:
                ag.observe(env, acts)
        k = r.result_type if r is not None else "HANG"
        if k == "ryuukyoku": ryu += 1
        elif k != "HANG":
            wins += 1
            if k == "ron": dealin += 1
        for ag in ags:
            for kk, vv in ag.calls.items():
                calls[kk] += vv
    N = 4 * a.deals
    print(f"hybrid_lm_call({a.model}): {a.deals}局 每座位和牌 {100*wins/N:.2f}% "
          f"每座位放铳 {100*dealin/N:.2f}% 流局 {100*ryu/a.deals:.1f}% "
          f"叫牌 {calls} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
