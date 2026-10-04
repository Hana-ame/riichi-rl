"""加载训练好的模型并跑一局，输出可读的牌谱。

用法：
    python play_model.py --ckpt checkpoints/riichi_net.pt --games 3
    python play_model.py --ckpt checkpoints/riichi_net.pt --greedy --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mahjong import env as E
from mahjong import obs2 as O2
from mahjong import tiles as T
from mahjong.net import MahjongNet
from train_reinforce import slot_of, N_ACTIONS


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = MahjongNet().to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(f"loaded {path}  arch={ck.get('arch')}  "
          f"iters={ck.get('iters')}")
    return model


def play(model, encoder, seed, device, temperature=1.0, greedy=True,
         verbose=True):
    env = E.MahjongEnv(seed=seed, dealer=seed % 4)
    result = env.start_turn()
    steps = 0
    while result is None and steps < 400:
        steps += 1
        actions = {}
        for p in env.pending_players():
            planes, scalar = encoder.encode(env, p)
            vec = np.concatenate([planes.ravel(), scalar]).astype(np.float32)
            legal = env.legal_actions(p)
            if not legal:
                actions[p] = E.PASS_ACTION
                continue
            slots = [slot_of(a) for a in legal]
            with torch.no_grad():
                x = torch.from_numpy(vec).unsqueeze(0).to(device)
                logits = model(x)[0]
            mask = torch.full((N_ACTIONS,), -1e9, dtype=torch.float32)
            mask[slots] = 0.0
            logits = logits + mask.to(device)
            if greedy:
                pick = int(logits.argmax())
            else:
                probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
                pick = int(torch.multinomial(probs, 1))
            if pick not in slots:
                pick = slots[0]
            act = legal[slots.index(pick)]
            actions[p] = act
            if verbose and env.phase == E.Phase.DISCARD:
                tag = {E.ACT_TSUMO: "和了!", E.ACT_RIICHI: "立直",
                       E.ACT_ANKAN: "暗杠", E.ACT_RYUUKYOKU: "流局宣言",
                       }.get(act.type, "")
                hand = T.counts_str(env.players[p].counts())
                if tag:
                    print(f"  [P{p}] {tag} 手牌={hand}")
        result = env.step(actions)
    return env, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/riichi_net.pt")
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--sample", action="store_true",
                    help="采样而非贪心")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, device)
    encoder = O2.ObsEncoder()

    wins = draws = 0
    for g in range(args.games):
        seed = args.seed if args.seed is not None else 1000 + g
        env, result = play(model, encoder, seed, device,
                           greedy=not args.sample,
                           verbose=not args.quiet)
        if result is None:
            print(f"game {g} seed={seed}: 未结束")
            continue
        if result.result_type == "ryuukyoku":
            draws += 1
            print(f"game {g} seed={seed}: 流局  "
                  f"点数差={result.points_delta}")
        else:
            wins += 1
            print(f"game {g} seed={seed}: {result.result_type} "
                  f"winner={result.winner} from={result.from_player} "
                  f"yaku={result.yaku} han={result.han} fu={result.fu} "
                  f"delta={result.points_delta}")

    print(f"\n共 {args.games} 局：和了 {wins}，流局 {draws}")


if __name__ == "__main__":
    main()
