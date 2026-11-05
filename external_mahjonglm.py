"""外部模型 mahjonglm-10m/100m 适配：Qwen3 词级 LM（828 词表，mjai 风格事件 token）。

策略：弃牌决策用模型（在完整视图 prompt 上取合法 discard token 的最高 logit）；
吃碰杠/立直/和牌的规则由本地规则接管（规则优先），保证合法对局。

用法：
  from external_mahjonglm import MahjongLM10Agent
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from mahjong import env as E
from mahjong import policies as POL
from mahjong import refengine as REF

ROOT = os.path.dirname(os.path.abspath(__file__))
TILE_NAMES = [f"m{i+1}" for i in range(9)] + [f"p{i+1}" for i in range(9)] \
             + [f"s{i+1}" for i in range(9)] + [f"z{i+1}" for i in range(7)]


class RuleFallback:
    """非弃牌决策（吃碰杠/立直/和牌）的规则接管。"""
    def __init__(self):
        self.riichi = POL.make_policy("riichi_needed", seed=0)
    def act(self, env, p, legal):
        for a in legal:
            if a.type in (E.ACT_TSUMO, E.ACT_RON):
                return a
        # 立直：能立且不是 already? 简化：合法就立
        for a in legal:
            if a.type == E.ACT_RIICHI:
                return a
        # 叫牌：默认 pass（避免独立叫牌头的负收益）
        for a in legal:
            if a.type == E.ACT_PASS:
                return a
        return legal[0]


class MahjongLMAgent:
    def __init__(self, model_dir, seat_name="mahjonglm10", max_tokens=4000):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
        self.model.eval()
        self.name = seat_name
        self.fallback = RuleFallback()
        self.hstate = None
        self.tokens = []       # 当前半庄/局的 prompt token 列表
        self.deal_idx = -1
        self.haipai_added = False
        self.max_tokens = max_tokens
        self.enc = self.tok

    def set_game(self, env, hstate):
        self.hstate = hstate
        self.tokens = ["<bos>", "rule_player_4", "rule_length_hanchan", "view_complete", "game_start"]
        self.deal_idx = -1
        self.haipai_added = False

    # ---- prompt helpers ----
    def _start_deal(self, env):
        dora = env.dora_indicators[0] if env.dora_indicators else 0
        self.tokens += ["round_start", f"bakaze_{(env.round_wind - 27) % 4}",
                        f"kyoku_{min(3, env.kyoku)}", "honba", "TENBO_ZERO",
                        "riichi_sticks", "TENBO_ZERO", "dora", TILE_NAMES[dora]]
        # 各家起手（庄家去掉刚摸的牌到 13 张）
        for i, p in enumerate(env.players):
            hand = list(p.hand)
            if len(hand) == 14:
                if p.last_drawn in hand:
                    hand.remove(p.last_drawn)
                else:
                    hand = hand[:13]
            hand = hand[:13]
            self.tokens.append(f"haipai_{i}")
            self.tokens += [TILE_NAMES[t] for t in hand]
        self.deal_idx = self.hstate.current_index() if self.hstate is not None else -1
        self.haipai_added = True

    def observe(self, env, acts):
        """记录本步产生的动作事件（discard/call/riichi），供下一步 prompt。"""
        # 若换局，重新开 prompt（只保留本半庄内的历史会丢，简化为从本局开始）
        if not self.haipai_added:
            self._start_deal(env)
        for p, a in acts.items():
            if a.type == E.ACT_DISCARD:
                self.tokens.append(f"discard_{p}_{TILE_NAMES[a.tile]}_tedashi")
            elif a.type == E.ACT_CHI:
                self.tokens += [f"opt_react_{p}_chi", f"take_react_{p}_chi"]
            elif a.type == E.ACT_PON:
                self.tokens += [f"opt_react_{p}_pon", f"take_react_{p}_pon"]
            elif a.type in (E.ACT_MINKAN, E.ACT_ANKAN, E.ACT_ADD_KAN):
                self.tokens += [f"opt_react_{p}_minkan", f"take_react_{p}_minkan"]
            elif a.type == E.ACT_RIICHI:
                self.tokens += [f"opt_self_{p}_riichi", f"take_self_{p}_riichi"]
            elif a.type in (E.ACT_TSUMO, E.ACT_RON):
                pass  # 和牌结束本局
        if len(self.tokens) > self.max_tokens:
            # 超长：截断到最近的事件（保留前缀）
            self.tokens = self.tokens[:5] + self.tokens[-(self.max_tokens - 5):]

    def _logits_next(self):
        ids = self.tok(" ".join(self.tokens), return_tensors="pt")["input_ids"]
        with torch.no_grad():
            out = self.model(ids)
        return out.logits[0, -1]

    def act(self, env, p, legal):
        if not self.haipai_added:
            self._start_deal(env)
        discards = [a for a in legal if a.type == E.ACT_DISCARD]
        if not discards:
            return self.fallback.act(env, p, legal)
        # 弃牌：候选 token（tsumogiri/tedashi）
        cand = []
        for a in discards:
            name = TILE_NAMES[a.tile]
            for suf in ("_tsumogiri", "_tedashi"):
                toks = self.tok.tokenize(f"discard_{p}_{name}{suf}")
                # 单 token 才纳入
                ids = self.tok.convert_tokens_to_ids([t for t in toks if t in self.tok.get_vocab()])
                if len(toks) == 1 and toks[0] in self.tok.get_vocab():
                    cand.append((float(self._logits_next()[self.tok.convert_tokens_to_ids(toks[0])]), a))
        if not cand:
            return self.fallback.act(env, p, legal)
        cand.sort(key=lambda x: -x[0])
        return cand[0][1]
