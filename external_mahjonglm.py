"""外部模型 mahjonglm-10m/100m 适配 v2（纯 torch，无 transformers 依赖）。

思路：按官方 tokenizer 的 mjai 词级语法，把当前对局拼成**忠实 prompt**：
  <bos> rule_player_4 rule_length_hanchan view_imperfect_{seat} game_start
  round_start bakaze_k kyoku_k honba <TENBO..> riichi_sticks <TENBO..> dora <tile>
  score_{i} TENBO_PLUS <面值拆分> ×4
  haipai_{seat} <13 张> / hidden_haipai_{i} ×3
  draw_{i}_{tile|hidden} discard_{i}_{tile}_{tsumogiri|tedashi} opt_*/take_*/pass_*
决策全部由 LM 的 next-token logits 给出：
  弃牌 = 合法 discard token 最大 logit；
  立直 = logit(opt_self_p_riichi) 高于最佳弃牌，且条件 take>pass；
  吃/碰/明杠 = 条件 take_react>pass_react；
  暗杠/加杠 = 条件 take_self>pass_self；
  和牌（自摸/荣和）= 规则必和（各家模型同口径）。
吃：take_react_p_chi 后按官方语法补 chi_pos_{low,mid,high} + red_not_used。
prompt 用 KV cache 增量喂入（每步只前向新 token），跨局重置。
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from mahjong import env as E
from mahjong import tiles as T
from mahjonglm_torch import load_model, load_tokenizer

TILE_TOK = ([f"m{i}" for i in range(1, 10)] + [f"p{i}" for i in range(1, 10)]
            + [f"s{i}" for i in range(1, 10)] + [f"z{i}" for i in range(1, 8)])
TENBO_VALS = [10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000,
              900, 800, 700, 600, 500, 400, 300, 200, 100]


def tile_tok(t):
    return TILE_TOK[int(t)]


def tenbo_split(n):
    """把点数拆成官方 TENBO_* 面值 token 序列（25000 -> 10000 10000 5000）。"""
    rem = int(abs(n)); out = []
    for v in TENBO_VALS:
        while rem >= v:
            out.append(f"TENBO_{v}"); rem -= v
    return out or ["TENBO_ZERO"]


class MahjongLMAgent:
    def __init__(self, model_dir, seat_name="mahjonglm", max_tokens=2048):
        self.model = load_model(model_dir)
        self.model.eval()
        self.enc, self.dec, self.vocab, self.inv = load_tokenizer(
            os.path.join(model_dir, "tokenizer.json"))
        self.name = seat_name
        self.max_tokens = max_tokens
        self.hstate = None
        self.tokens = []          # 当前局的 prompt token 串
        self.cache = None         # 每层 (k,v)
        self.fed = 0              # 已喂入 cache 的 token 数
        self.last_logits = None
        self.started = False
        self.seat = 0
        self.needs_draw = [False] * 4
        self.skip = set()         # 本步已由 act() 记录的座位
        self.decisions = {"discard": 0, "riichi": 0, "chi": 0, "pon": 0,
                          "kan": 0, "minkan": 0, "kakan": 0, "pass_call": 0}

    # ---------------- prompt 维护 ----------------
    def set_game(self, env, hstate=None):
        self.hstate = hstate
        self.tokens = []
        self.cache = None
        self.fed = 0
        self.last_logits = None
        self.started = False
        self.needs_draw = [False] * 4
        self.skip = set()

    def _start_deal(self, env, seat):
        self.seat = seat
        p = env.players[seat]
        hand = list(p.hand)
        if len(hand) == 14:
            if p.last_drawn in hand:
                hand.remove(p.last_drawn)
            else:
                hand = hand[:13]
        hand = sorted(hand)[:13]
        w = max(0, min(2, int(env.round_wind) - 27))
        tok = ["<bos>", "rule_player_4", "rule_length_hanchan",
               f"view_imperfect_{seat}", "game_start", "round_start",
               f"bakaze_{w}", f"kyoku_{max(0, min(3, int(env.kyoku) - 1))}",
               "honba"]
        tok += (["TENBO_ZERO"] if not env.honba else tenbo_split(int(env.honba) * 100))
        tok += ["riichi_sticks"]
        tok += (["TENBO_ZERO"] if not env.kyokuten else tenbo_split(int(env.kyokuten) * 1000))
        ind = env.dora_indicators[0] if env.dora_indicators else 0
        tok += ["dora", tile_tok(ind)]
        for i in range(4):
            pts = int(env.players[i].points)
            tok.append(f"score_{i}")
            tok.append("TENBO_PLUS" if pts >= 0 else "TENBO_MINUS")
            tok += tenbo_split(pts)
        for i in range(4):
            if i == seat:
                tok += [f"haipai_{i}"] + [tile_tok(t) for t in hand]
            else:
                tok += [f"hidden_haipai_{i}"]
        self.tokens = tok
        self.started = True

    def _ids(self, toks):
        return [self.vocab.get(t, 1) for t in toks]

    def _sync(self):
        """把未喂入的 token 增量前向，返回最后位置的 logits。"""
        if self.fed < len(self.tokens):
            new = self.tokens[self.fed:]
            logits, self.cache = self.model(
                torch.tensor([self._ids(new)]), self.cache)
            self.last_logits = logits[0, -1]
            self.fed = len(self.tokens)
        return self.last_logits

    def _push(self, toks):
        snap = (len(self.tokens), self.cache, self.last_logits)
        self.tokens.extend(toks)
        return snap

    def _restore(self, snap):
        n, cache, logits = snap
        del self.tokens[n:]
        self.cache = cache
        self.last_logits = logits
        self.fed = n

    def _lg(self, logits, tok):
        i = self.vocab.get(tok)
        if i is None:
            return -1e9
        return float(logits[i])

    def _discard_choice(self, logits, p, discards, extra_legal=()):
        p_last = getattr(self, "_last_drawn", -1)
        best, best_s = None, -1e18
        for a in discards:
            nm = tile_tok(a.tile)
            cand = []
            if a.tile == p_last:
                cand.append(f"discard_{p}_{nm}_tsumogiri")
            else:
                cand.append(f"discard_{p}_{nm}_tedashi")
            for t in cand:
                s = self._lg(logits, t)
                if s > best_s:
                    best_s, best = s, (a, t)
        return best

    # ---------------- 事件记录 ----------------
    def _record_draw(self, p, tile, own):
        if own:
            self.tokens.append(f"draw_{p}_{tile_tok(tile)}")
        else:
            self.tokens.append(f"draw_{p}_hidden")

    def observe(self, env, acts):
        if not self.started:
            self._start_deal(env, self.seat)
        # 本步先补对方的摸牌事件
        for p, a in sorted(acts.items(), key=lambda kv: kv[0]):
            if p in self.skip:
                continue
            own = (p == self.seat)
            if a.type == E.ACT_DISCARD:
                if self.needs_draw[p]:
                    self._record_draw(p, env.players[p].last_drawn, own)
                    self.needs_draw[p] = False
                nm = tile_tok(a.tile)
                suf = "tsumogiri" if env.players[p].last_drawn == a.tile else "tedashi"
                self.tokens.append(f"discard_{p}_{nm}_{suf}")
                self.needs_draw = [False] * 4
                self.needs_draw[(p + 1) % 4] = True
            elif a.type == E.ACT_CHI:
                called = a.arg if a.arg >= 0 else a.tile
                pos = "low" if called == a.tile else ("mid" if called == a.tile + 1 else "high")
                self.tokens += [f"opt_react_{p}_chi", f"take_react_{p}_chi",
                                f"chi_pos_{pos}", "red_not_used"]
                self.needs_draw = [False] * 4
            elif a.type == E.ACT_PON:
                self.tokens += [f"opt_react_{p}_pon", f"take_react_{p}_pon"]
                self.needs_draw = [False] * 4
            elif a.type == E.ACT_MINKAN:
                self.tokens += [f"opt_react_{p}_minkan", f"take_react_{p}_minkan"]
                self.needs_draw = [False] * 4
                self.needs_draw[p] = True
            elif a.type in (E.ACT_ANKAN, E.ACT_ADD_KAN):
                kind = "ankan" if a.type == E.ACT_ANKAN else "kakan"
                self.tokens += [f"opt_self_{p}_{kind}", f"take_self_{p}_{kind}"]
                self.needs_draw[p] = True
            elif a.type == E.ACT_RIICHI:
                pass  # act() 已记录
            elif a.type in (E.ACT_TSUMO, E.ACT_RON):
                self.tokens += [f"opt_self_{p}_tsumo" if a.type == E.ACT_TSUMO
                                else f"opt_self_{p}_ron",
                                f"take_self_{p}_tsumo" if a.type == E.ACT_TSUMO
                                else f"take_self_{p}_ron"]
        self.skip = set()
        if len(self.tokens) > self.max_tokens:
            # 超长：丢掉最老的事件，保留前缀与尾部（cache 需重建）
            keep = self.max_tokens
            self.tokens = self.tokens[:18] + self.tokens[-(keep - 18):]
            self.cache = None; self.fed = 0; self.last_logits = None

    # ---------------- 决策 ----------------
    def act(self, env, p, legal):
        if not self.started:
            self._start_deal(env, p)
        self._last_drawn = env.players[p].last_drawn
        types = {a.type for a in legal}
        # 和牌：规则必和（与自研模型同口径）
        for wt in (E.ACT_TSUMO, E.ACT_RON):
            if wt in types:
                tok = ["opt_self_%d_tsumo" % p, "take_self_%d_tsumo" % p] if wt == E.ACT_TSUMO \
                    else ["opt_react_%d_ron" % p, "take_react_%d_ron" % p]
                self.tokens += tok
                self.skip = {p}
                return next(a for a in legal if a.type == wt)

        discards = [a for a in legal if a.type == E.ACT_DISCARD]
        if discards:
            if self.needs_draw[p]:
                self._record_draw(p, self._last_drawn, True)
                self.needs_draw[p] = False
            return self._self_decision(env, p, legal, discards)

        # 响应阶段（吃碰杠/过）
        return self._response_decision(env, p, legal)

    def _commit_discard(self, p, a, tok):
        self.tokens.append(tok)
        self.needs_draw = [False] * 4
        self.needs_draw[(p + 1) % 4] = True
        self.decisions["discard"] += 1
        self.skip = {p}
        return a

    def _self_decision(self, env, p, legal, discards):
        L = self._sync()
        riichis = [a for a in legal if a.type == E.ACT_RIICHI]
        kans = [a for a in legal if a.type in (E.ACT_ANKAN, E.ACT_ADD_KAN)]
        best_disc = self._discard_choice(L, p, discards)
        best_s = self._lg(L, best_disc[1])

        # 立直：比较 opt_self_riichi 与最佳弃牌 logit
        if riichis:
            s_opt = self._lg(L, f"opt_self_{p}_riichi")
            if s_opt > best_s:
                snap = self._push([f"opt_self_{p}_riichi"])
                L1 = self._sync()
                s_take = self._lg(L1, f"take_self_{p}_riichi")
                s_pass = self._lg(L1, f"pass_self_{p}_riichi")
                if s_take >= s_pass:
                    self.tokens.append(f"take_self_{p}_riichi")
                    L2 = self._sync()
                    best, bt = None, None
                    for a in riichis:
                        nm = tile_tok(a.tile)
                        tok = f"discard_{p}_{nm}_tsumogiri" if a.tile == self._last_drawn \
                            else f"discard_{p}_{nm}_tedashi"
                        s = self._lg(L2, tok)
                        if best is None or s > best_s:
                            best_s, best, bt = s, a, tok
                    if best is not None:
                        self.tokens.append(bt)
                        self.needs_draw = [False] * 4
                        self.needs_draw[(p + 1) % 4] = True
                        self.decisions["riichi"] += 1
                        self.skip = {p}
                        return best
                self._restore(snap)

        # 暗杠/加杠
        for a in kans:
            kind = "ankan" if a.type == E.ACT_ANKAN else "kakan"
            s_opt = self._lg(L, f"opt_self_{p}_{kind}")
            if s_opt < best_s - 2.0:
                continue
            snap = self._push([f"opt_self_{p}_{kind}"])
            L1 = self._sync()
            s_take = self._lg(L1, f"take_self_{p}_{kind}")
            s_pass = self._lg(L1, f"pass_self_{p}_{kind}")
            if s_take >= s_pass:
                self.tokens.append(f"take_self_{p}_{kind}")
                self.needs_draw = [False] * 4
                self.needs_draw[p] = True
                self.decisions["kan"] += 1
                self.skip = {p}
                return a
            self._restore(snap)

        a, tok = best_disc
        return self._commit_discard(p, a, tok)

    def _response_decision(self, env, p, legal):
        types = {a.type for a in legal}
        if types == {E.ACT_PASS} or len(legal) == 1:
            return E.PASS_ACTION if E.PASS_ACTION in legal else legal[0]
        L = self._sync()
        cands = []
        for a in legal:
            if a.type == E.ACT_CHI:
                cands.append(("chi", a))
            elif a.type == E.ACT_PON:
                cands.append(("pon", a))
            elif a.type in (E.ACT_MINKAN,):
                cands.append(("minkan", a))
            elif a.type == E.ACT_ADD_KAN:
                cands.append(("kakan", a))
        best = None
        for kind, a in cands:
            if kind in ("chi", "pon", "minkan"):
                opt = f"opt_react_{p}_{kind}"
                take = f"take_react_{p}_{kind}"
                pas = f"pass_react_{p}_{kind}_voluntary"
            else:
                opt = f"opt_self_{p}_{kind}"
                take = f"take_self_{p}_{kind}"
                pas = f"pass_self_{p}_{kind}"
            if self._lg(L, opt) < -1e8:
                continue
            snap = self._push([opt])
            L1 = self._sync()
            margin = self._lg(L1, take) - self._lg(L1, pas)
            self._restore(snap)
            if best is None or margin > best[0]:
                best = (margin, kind, a, opt, take, pas)
        if best is not None and best[0] > 0:
            margin, kind, a, opt, take, pas = best
            self.tokens += [opt, take]
            if kind == "chi":
                called = a.arg if a.arg >= 0 else a.tile
                pos = "low" if called == a.tile else ("mid" if called == a.tile + 1 else "high")
                self.tokens += [f"chi_pos_{pos}", "red_not_used"]
                self.needs_draw = [False] * 4
                self.decisions["chi"] += 1
            elif kind in ("pon", "minkan"):
                if kind == "minkan":
                    self.needs_draw = [False] * 4
                    self.needs_draw[p] = True
                else:
                    self.needs_draw = [False] * 4
                self.decisions[kind] += 1
            else:
                self.needs_draw[p] = True
                self.decisions["kan"] += 1
            self.skip = {p}
            return a
        # 过：为所有被提供的选项写入 pass 记录（官方语法）
        for kind, a in cands:
            if kind in ("chi", "pon", "minkan"):
                self.tokens += [f"opt_react_{p}_{kind}",
                                f"pass_react_{p}_{kind}_voluntary"]
            else:
                self.tokens += [f"opt_self_{p}_{kind}", f"pass_self_{p}_{kind}"]
        self.decisions["pass_call"] += 1
        self.skip = {p}
        return E.PASS_ACTION if E.PASS_ACTION in legal else legal[0]


def make_agent(model_dir="external/mahjonglm10", name="mahjonglm"):
    return MahjongLMAgent(model_dir, seat_name=name)
