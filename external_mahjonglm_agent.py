"""MahjongLM 作为外部对手（最小可用版：弃牌由 LM 打分，其余规则兜底）。

思路：把当前可见局面拼成 MahjongLM 的 token 前缀，然后在
`discard_{seat}_{tile}_{tedashi|tsumogiri}` 合法 token 上用 logits argmax 选牌；
立直/和牌/吃碰杠用规则（CallFirst/可和即和）兜底，避免动作与 LM 语法不匹配。
"""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from mahjong import env as E
from mahjong.hanchan import HanchanState
from mahjong.tiles import NUM_TILES
from external_mahjonglm import load_model, load_tokenizer

TILE34 = [*("m%d" % i for i in range(1, 10)), *("p%d" % i for i in range(1, 10)),
          *("s%d" % i for i in range(1, 10)), "E", "S", "W", "N", "P", "F", "C"]
RED = ["m0", "p0", "s0"]

def tile_token(t):
    return TILE34[int(t) % 34]

class MahjongLMAgent:
    def __init__(self, name="mahjonglm10", ckpt_dir="external/mahjonglm10"):
        self.name = name
        self.model = load_model(ckpt_dir)
        self.enc, self.dec, self.vocab, self.inv = load_tokenizer(f"{ckpt_dir}/tokenizer.json")
        self.hstate = None
        self.seat = 0

    def set_game(self, env, hstate):
        self.hstate = hstate

    def _prefix(self, env, p):
        h = self.hstate
        toks = ["<bos>", "rule_player_4", "rule_length_hanchan",
                f"view_imperfect_{p}", "game_start", "round_start",
                f"bakaze_{(env.round_wind - 27) if env.round_wind is not None else 0}",
                f"kyoku_{env.kyoku}", "honba"] + ["TENBO_ZERO"] * max(0, env.honba) + \
               ["riichi_sticks", "TENBO_ZERO"]
        if env.dora_indicator is not None:
            toks += ["dora", tile_token(env.dora_indicator)]
        for i in range(4):
            toks += [f"score_{i}", "TENBO_PLUS", "TENBO_10000", "TENBO_10000", "TENBO_5000"]
        hand = sorted(env.players[p].counts())
        haipai = []
        for t, c in enumerate(hand):
            haipai += [tile_token(t)] * min(c, 4)
        toks += [f"haipai_{p}"] + haipai[:13]
        for i in range(4):
            q = env.players[i]
            if not q.river and not q.melds:
                continue
            for t in q.river:
                toks += [f"discard_{i}_{tile_token(t)}_tedashi"]
        return toks

    def act(self, env, p, legal):
        discards = [a for a in legal if a.type == E.ACT_DISCARD]
        if not discards:
            # 和牌/立直/吃碰杠：规则兜底
            for kind in (E.ACT_TSUMO, E.ACT_RON):
                for a in legal:
                    if a.type == kind: return a
            return E.PASS_ACTION
        toks = self._prefix(env, p) + [f"draw_{p}_" + tile_token(env.players[p].last_drawn
                                     if env.players[p].last_drawn >= 0 else 0)]
        ids = self.enc(" ".join(toks))
        with torch.no_grad():
            logits = self.model(torch.tensor([ids]))[0, -1]
        best, best_s = None, -1e18
        for a in discards:
            tt = tile_token(a.tile)
            for kind in ("tedashi", "tsumogiri"):
                tok = f"discard_{p}_{tt}_{kind}"
                if tok in self.vocab:
                    s = float(logits[self.vocab[tok]])
                    if s > best_s:
                        best_s, best = s, a
        return best if best is not None else discards[0]
