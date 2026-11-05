"""外部模型蒸馏数据采集：4 席都由 mahjonglm（10m/100m）决策，录 (obs, 47 槽标签)。

与 gen_teacher_data.py 同口径（obs2 + touch + 合法槽 mask + 标签槽 + dora aux），
教师换成外部 LM（弃牌/立直/吃碰杠都是 LM logits 决策，见 external_mahjonglm.py）。
产物可直接喂 train_teacher_bc.py（可多文件拼接）。

用法：
  python distill_external.py --model external/mahjonglm10 --name mjlm10 \
      --hanchans 40 --out data/distill_mjlm10.npz
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from mahjong import env as E
from mahjong import obs2 as O2
from mahjong.hanchan import HanchanState
from mahjong import policies as POL
from train_hanchan import encode, slot_of, N_ACTIONS
from external_mahjonglm import MahjongLMAgent


def run_hanchan(enc, h, agents, seed0):
    xs, tfs, masks, labels, aux = [], [], [], [], []
    deals = 0
    while not h.finished:
        env = h.new_env(seed=seed0 + h.current_index() * 131 + deals)
        for ag in agents:
            ag.set_game(env, h)
        deal_ps = []
        r = env.start_turn()
        steps = 0
        while r is None and steps < 400:
            steps += 1
            acts = {}
            for p in env.pending_players():
                legal = env.legal_actions(p)
                if not legal:
                    acts[p] = E.PASS_ACTION
                    continue
                vec, sh, sit, gu, tf = encode(env, enc, h, p)
                slots = [slot_of(a) for a in legal]
                if any(s < 0 or s >= N_ACTIONS for s in slots):
                    acts[p] = E.PASS_ACTION if E.PASS_ACTION in legal else legal[0]
                    continue
                act = agents[p].act(env, p, legal)
                if act not in legal:
                    act = E.PASS_ACTION if E.PASS_ACTION in legal else legal[0]
                lab = slot_of(act)
                if lab not in slots:
                    lab = slots[0]
                m = np.zeros(N_ACTIONS, dtype=bool)
                m[slots] = True
                xs.append(vec); tfs.append(tf); masks.append(m); labels.append(lab)
                deal_ps.append(p)
                acts[p] = act
            r = env.step(acts)
            for ag in agents:
                ag.observe(env, acts)
        if r is None:
            aux.extend([0.0] * len(deal_ps))
            break
        won = (r.result_type not in ("ryuukyoku",)) and getattr(r, "winner", None) is not None
        has_dora = (getattr(r, "dora_count", 0) + getattr(r, "aka_count", 0)
                    + getattr(r, "ura_dora_count", 0)) > 0
        for p0 in deal_ps:
            aux.append(1.0 if (won and r.winner == p0 and has_dora) else 0.0)
        deals += 1
        h.advance(r)
    return (np.stack(xs), np.stack(tfs).reshape(len(xs), -1),
            np.stack(masks), np.array(labels, dtype=np.int64),
            np.array(aux, dtype=np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="external/mahjonglm10")
    ap.add_argument("--name", default="mjlm10")
    ap.add_argument("--hanchans", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", default="data/distill_mjlm10.npz")
    args = ap.parse_args()
    torch.set_num_threads(2)
    enc = O2.ObsEncoder()
    agents = [MahjongLMAgent(args.model, seat_name=f"{args.name}_{i}") for i in range(4)]
    t0 = time.time()
    parts = []
    for i in range(args.hanchans):
        h = HanchanState(dealer=i % 4)
        parts.append(run_hanchan(enc, h, agents, 20_000 + i * 7919))
        print(f"  {i+1}/{args.hanchans} 半庄 {sum(len(p[3]) for p in parts)} 样本 "
              f"({time.time()-t0:.0f}s)", flush=True)
    X = np.concatenate([p[0] for p in parts]); TF = np.concatenate([p[1] for p in parts])
    M = np.concatenate([p[2] for p in parts]); Y = np.concatenate([p[3] for p in parts])
    AUX = np.concatenate([p[4] for p in parts])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, x=X, tf=TF, mask=M, y=Y, aux=AUX)
    print(f"样本 {len(X)} 观测 {X.shape[1]} touch {TF.shape[1]} 槽 {M.shape[1]}")
    for name, slot in (("tsumo", 34), ("ron", 35), ("riichi", 36)):
        n = int((Y == slot).sum()); print(f"  {name:>6}: {n} ({100.0*n/len(Y):.2f}%)")
    print(f"写入 {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
