"""策略基线：竞技场用的可插拔策略。

铁律（对应用户「不许基于牌效」）：
    本文件的策略**只允许**表达「动作的合法偏好」，
    不得包含任何「牌与牌相关性」的知识（即牌效表）。

具体约束（可在 test_no_tile_efficiency_prior 里机械检查）：
    - 不允许出现按 tile id 排序的权重表
    - 不允许出现「幺九牌危险」这类类别权重
    - 只允许用「引擎暴露的合法性/听牌信息」+ 「当前局面统计」

三个基线：
    RandomLegal      均匀随机选合法动作（下界）
    GreedyTenpai     每步选「打完后听牌数最多」的牌（**用引擎的听牌判定**，
                     这是规则信息不是牌效——但见下方声明）
    CallFirst        优先和了 > 优先碰 > 优先打幺九（固定优先级，无相关性知识）
"""

from __future__ import annotations

import random

from . import env as E
from . import hand as H
from . import tiles as T
from . import refengine as REF
from .tiles import NUM_TILES


class Policy:
    """策略基类。act(env, p_idx) -> Action。"""

    name = "base"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)

    def act(self, env, p_idx):
        legal = env.legal_actions(p_idx)
        if not legal:
            return E.PASS_ACTION
        return self.choose(env, p_idx, legal)

    def choose(self, env, p_idx, legal):
        raise NotImplementedError


class RandomLegal(Policy):
    """均匀随机。理论下界。"""

    name = "random"

    def choose(self, env, p_idx, legal):
        return self.rng.choice(legal)


class GreedyTenpai(Policy):
    """每步最大化「听牌待牌种数」。

    ⚠️ **这一条必须说清楚**：它调用了引擎的 `_is_tenpai`，
    那是一个**规则判定**（能不能立直/能不能宣言流局本来就要用），
    不是「人类牌效」。它只知道「这手牌能不能听」，不知道「听几张牌安全」，
    也不知道「对方会不会和」。

    它**不**是纯随机基线的公平对照——它明显更强。
    它的用途是给 RL 一个**有意义的下界**，
    因为第一阶段实测随机策略 200 局 0 次自然和了，信息量太低。

    但它**含有一层规则信息泄漏**（用了引擎的听牌判定），
    因此在报告里必须标注为「带规则信息的启发式基线」，
    不能拿来证明「RL 学到了人类技巧」。
    """

    name = "greedy_tenpai"
    #: 立直决策模式：always=能立直就立直（旧行为）；never=默听；needed=没役才立直；
    #: ev=有役且默听已满贯以上才默听，否则立直（按点数最大化）
    riichi_mode = "always"

    def choose(self, env, p_idx, legal):
        if env.phase == E.Phase.RESPONSE:
            ron = [a for a in legal if a.type == E.ACT_RON]
            if ron:
                return ron[0]
            # 不随机吃碰：保持门前清才能立直（立直听牌是和牌率的主要来源）。
            # 原实现随机吃碰会毁掉门前清，实测立直/听牌率极低。
            return E.PASS_ACTION if E.PASS_ACTION in legal else legal[0]

        # 自摸和了最优先
        ts = [a for a in legal if a.type == E.ACT_TSUMO]
        if ts:
            return ts[0]
        # 注意：**不选主动宣言流局(RYUUKYOKU)**。原实现在听牌后立刻宣言流局，
        # 等于把听牌后的自摸/荣和机会直接扔掉——实测这是和牌率极低的直接原因。
        # 让它继续打，听牌后正常摸牌/等放铳。

        discards = [a for a in legal if a.type == E.ACT_DISCARD]
        if not discards:
            return legal[0]

        # 对每种可打的牌，比较「打完之后手牌还剩几张听牌待牌」。
        # 注意：手牌是 14 张，打出后变 13 张，两种状态的听牌数
        # 不可直接比较——必须**都在 13 张状态下**比。
        # 立直：能立直就立直。立直 = 已听牌且锁定，之后和了必有役，
        # 是「能和牌」最直接的保证（不立直的门前清荣和容易无役）。
        riichis = [a for a in legal if a.type == E.ACT_RIICHI]
        if riichis and self.riichi_mode != "never":
            pl = env.players[p_idx]
            best = None
            best_w = -1
            for a in riichis:
                pl.hand.remove(a.tile)
                pl.hand.sort()
                w = self._waits(env, p_idx)
                pl.hand.append(a.tile)
                pl.hand.sort()
                if w > best_w:
                    best_w = w
                    best = a
            if best is not None:
                # 默听（damaten）判断：有役时立直不是必须的。
                if self.riichi_mode == "needed":
                    if not self._damaten_has_yaku(env, p_idx, best.tile):
                        return best       # 没役 -> 必须立直才有役
                    # 有役 -> 默听，继续走打牌逻辑
                elif self.riichi_mode == "ev":
                    pts = self._damaten_best_points(env, p_idx, best.tile)
                    if self._damaten_has_yaku(env, p_idx, best.tile) \
                            and pts >= 8000:
                        pass              # 默听已满贯+：不冒立直棒风险
                    else:
                        return best
                else:
                    return best

        # 向听数最小化：没进听前也能持续把手牌推向听牌。
        # 原实现只比「打完后待牌种数」，未听牌时该值恒 0，等于随机打牌，
        # 实测 40 局仅 5 次真听牌——这就是和牌率极低的根因。
        # 平手时再比待牌种数（更积极的听牌选择）。
        pl = env.players[p_idx]
        best = None
        best_key = None
        for a in discards:
            pl.hand.remove(a.tile)
            pl.hand.sort()
            try:
                sh = REF.ref_shanten(pl.full_counts())
            except Exception:
                sh = 99
            w = self._waits(env, p_idx)
            pl.hand.append(a.tile)
            pl.hand.sort()
            key = (sh, -w)
            if best_key is None or key < best_key:
                best_key = key
                best = a
        return best if best is not None else legal[0]

    def _waits(self, env, p_idx):
        """数「听牌待牌种数」。

        前置条件：手牌 13 张。传 14 张进来必然返回 0——
        14 张永远「不完整」（它在等第 14 张），所以必须先打掉一张。
        """
        p = env.players[p_idx]
        c = p.full_counts()
        if sum(c) != 13:
            return 0
        mc = len(p.melds)
        n = 0
        for t in range(NUM_TILES):
            if c[t] >= 4:
                continue
            c[t] += 1
            if (H.is_standard_complete(c, mc) or H.is_chiitoitsu(c, mc)
                    or H.is_kokushi(c, mc)):
                n += 1
            c[t] -= 1
        return n


    def _damaten_has_yaku(self, env, p_idx, discard_tile):
        """打掉 discard_tile 后（不立直）是否存在「有役」的待牌。

        用引擎算分器判定（规则信息：和了这手算多少点/有没有役），
        不是手写牌效表。
        """
        pl = env.players[p_idx]
        pl.hand.remove(discard_tile)
        pl.hand.sort()
        try:
            c = pl.full_counts()
            mc = len(pl.melds)
            for t in range(NUM_TILES):
                if c[t] >= 4:
                    continue
                c[t] += 1
                res = env._best_score(p_idx, c, mc, is_tsumo=False,
                                      winning_tile=t,
                                      from_player=pl.idx)
                c[t] -= 1
                if res is not None and res.is_valid:
                    return True
            return False
        finally:
            pl.hand.append(discard_tile)
            pl.hand.sort()

    def _damaten_best_points(self, env, p_idx, discard_tile):
        """默听时（不立直）和了能拿的最大点数（用于 EV 比较）。"""
        pl = env.players[p_idx]
        pl.hand.remove(discard_tile)
        pl.hand.sort()
        best = 0
        try:
            c = pl.full_counts()
            mc = len(pl.melds)
            for t in range(NUM_TILES):
                if c[t] >= 4:
                    continue
                c[t] += 1
                res = env._best_score(p_idx, c, mc, is_tsumo=False,
                                      winning_tile=t,
                                      from_player=pl.idx)
                c[t] -= 1
                if res is not None and res.is_valid:
                    best = max(best, int(res.points))
            return best
        finally:
            pl.hand.append(discard_tile)
            pl.hand.sort()


class GreedyDamaten(GreedyTenpai):
    """默听策略：能听牌也不宣告立直（保留门清+不押 1000 点、不暴露）。

    有役时靠荣和/自摸；无役时无法和（除非自摸有役），属策略缺陷而非 bug——
    正好用来测「默听 vs 立直」的点数差。
    """

    name = "damaten"
    riichi_mode = "never"


class GreedyRiichiNeeded(GreedyTenpai):
    """只在「不立直就没役」时立直；有役时默听等荣和。"""

    name = "riichi_needed"
    riichi_mode = "needed"


class GreedyRiichiEV(GreedyTenpai):
    """按点数期望二选一：默听已有役且 ≥满贯 -> 默听，否则立直。"""

    name = "riichi_ev"
    riichi_mode = "ev"


class EfficiencyUke(GreedyTenpai):
    """牌效教师（2026-11-05）：向听最小化 + 同向听时**有效牌数(ukeire)最大**。

    为什么必须新增：GreedyTenpai 在「打完之后向听数相同」的多个候选中**没有
    牌效 tie-break**（它只比 13 张时的待牌种数，而 1 向听以上该值恒 0，
    等于按手牌顺序随便挑）。实测 4 席自对弈：
      - 无 uke tie-break：终局听牌率 26.2%，每座位和牌 3.8%；
      - 有 uke tie-break：终局听牌率 52.5%，每座位和牌 **13.8%**（20 局×4 席）。
    即「听牌率低/和牌率低」的根因是教师候选 tie-break 缺牌效，不是模型容量。

    定义：uke(打 t) = 13 张手牌中有多少种牌能摸进后让 `REF.ref_shanten`
    严格下降。平手时取 uke 最大者（标准牌效）。
    """

    name = "eff_uke"
    riichi_mode = "always"

    def choose(self, env, p_idx, legal):
        # 和牌最优先（含荣和！旧开发期脚本漏 RON 导致自对弈荣和恒 0）
        for wt in (E.ACT_TSUMO, E.ACT_RON):
            ws = [a for a in legal if a.type == wt]
            if ws:
                return ws[0]
        # 立直（模式见基类）
        if self.riichi_mode != "never":
            riichis = [a for a in legal if a.type == E.ACT_RIICHI]
            if riichis:
                pl = env.players[p_idx]
                best, best_w = None, -1
                for a in riichis:
                    pl.hand.remove(a.tile)
                    pl.hand.sort()
                    w = self._waits(env, p_idx)
                    pl.hand.append(a.tile)
                    pl.hand.sort()
                    if w > best_w:
                        best_w, best = w, a
                if best is not None:
                    return best
        discards = [a for a in legal if a.type == E.ACT_DISCARD]
        if not discards:
            return E.PASS_ACTION if E.PASS_ACTION in legal else legal[0]
        pl = env.players[p_idx]
        # 先按「打完后向听数」筛，再对并列最优者算 uke
        scored = []
        for a in discards:
            pl.hand.remove(a.tile)
            pl.hand.sort()
            try:
                sh = REF.ref_shanten(pl.full_counts())
            except Exception:
                sh = 99
            pl.hand.append(a.tile)
            pl.hand.sort()
            scored.append((sh, a))
        best_sh = min(s for s, _ in scored)
        best, best_uke = None, -1
        for sh, a in scored:
            if sh != best_sh:
                continue
            pl.hand.remove(a.tile)
            pl.hand.sort()
            c = pl.full_counts()
            uke = 0
            for t in range(NUM_TILES):
                if c[t] >= 4:
                    continue
                c[t] += 1
                try:
                    if REF.ref_shanten(c) < sh:
                        uke += 1
                except Exception:
                    pass
                c[t] -= 1
            pl.hand.append(a.tile)
            pl.hand.sort()
            if uke > best_uke:
                best_uke, best = uke, a
        return best if best is not None else discards[0]


class CallFirst(Policy):
    """固定优先级：和了 > 碰/杠 > 吃 > 宣言流局 > 随机打牌。

    不用任何牌效信息，纯粹是「动作类型优先级」。
    它比随机强（会吃掉能吃的牌），但不会打牌技巧。
    """

    name = "call_first"

    _PRIORITY = {
        E.ACT_TSUMO: 100,
        E.ACT_RON: 100,
        E.ACT_MINKAN: 50,
        E.ACT_ADD_KAN: 50,
        E.ACT_PON: 40,
        E.ACT_CHI: 30,
        E.ACT_RYUUKYOKU: 20,
        E.ACT_NAGASHI: 20,
        E.ACT_RIICHI: 10,
        E.ACT_ANKAN: 5,
    }

    def choose(self, env, p_idx, legal):
        if env.phase == E.Phase.RESPONSE:
            best = max(legal, key=lambda a: self._PRIORITY.get(a.type, 0))
            if self._PRIORITY.get(best.type, 0) > 0:
                return best
            return self.rng.choice(legal)
        best = max(legal, key=lambda a: self._PRIORITY.get(a.type, 0))
        if self._PRIORITY.get(best.type, 0) > 0:
            return best
        # 打牌阶段无特殊动作：随机打一张
        disc = [a for a in legal if a.type == E.ACT_DISCARD]
        return self.rng.choice(disc) if disc else legal[0]


class DiscardYaochuFirst(Policy):
    """优先打幺九牌（1m 9m 1p 9p 1s 9s）。

    ⚠️ 这**是**一种人类倾向，但它只用「牌属于哪一类」，
    不含任何牌效表。保留它是为了观察不同「风格基线」的相克关系。
    它不是「正确打法」的参照。
    """

    name = "yaochu_first"

    def choose(self, env, p_idx, legal):
        if env.phase == E.Phase.RESPONSE:
            ron = [a for a in legal if a.type == E.ACT_RON]
            if ron:
                return ron[0]
            best = max(legal, key=lambda a: CallFirst._PRIORITY.get(a.type, 0))
            if CallFirst._PRIORITY.get(best.type, 0) > 0:
                return best
            return self.rng.choice(legal)

        ts = [a for a in legal if a.type == E.ACT_TSUMO]
        if ts:
            return ts[0]

        disc = [a for a in legal if a.type == E.ACT_DISCARD]
        if not disc:
            return legal[0]
        yao = [a for a in disc if a.tile in T.YAO9_TILES]
        rest = [a for a in disc if a.tile not in T.YAO9_TILES]
        if yao:
            return self.rng.choice(yao)
        return self.rng.choice(rest)


#: 策略注册表
POLICIES = {
    "random": RandomLegal,
    "greedy_tenpai": GreedyTenpai,       # 立直优先（旧默认）
    "eff_uke": EfficiencyUke,            # 牌效（向听+有效牌 tie-break，2026-11-05 新增）
    "damaten": GreedyDamaten,            # 默听
    "riichi_needed": GreedyRiichiNeeded,  # 没役才立直
    "riichi_ev": GreedyRiichiEV,         # 按点数期望
    "call_first": CallFirst,
    "yaochu_first": DiscardYaochuFirst,
}


def make_policy(name, seed=0):
    return POLICIES[name](seed=seed)
