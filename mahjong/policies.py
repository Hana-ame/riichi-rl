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

    def choose(self, env, p_idx, legal):
        if env.phase == E.Phase.RESPONSE:
            ron = [a for a in legal if a.type == E.ACT_RON]
            if ron:
                return ron[0]
            return self.rng.choice(legal)

        # 自摸和了最优先
        ts = [a for a in legal if a.type == E.ACT_TSUMO]
        if ts:
            return ts[0]
        ryuu = [a for a in legal if a.type == E.ACT_RYUUKYOKU]
        if ryuu:
            return ryuu[0]

        discards = [a for a in legal if a.type == E.ACT_DISCARD]
        if not discards:
            return legal[0]

        # 对每种可打的牌，比较「打完之后手牌还剩几张听牌待牌」。
        # 注意：手牌是 14 张，打出后变 13 张，两种状态的听牌数
        # 不可直接比较——必须**都在 13 张状态下**比。
        pl = env.players[p_idx]
        best = None
        best_score = -1
        for a in discards:
            pl.hand.remove(a.tile)
            pl.hand.sort()
            s = self._waits(env, p_idx)
            pl.hand.append(a.tile)
            pl.hand.sort()
            if s > best_score:
                best_score = s
                best = a
        return best if best is not None else legal[0]

    def _waits(self, env, p_idx):
        p = env.players[p_idx]
        c = p.full_counts()
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
    "greedy_tenpai": GreedyTenpai,
    "call_first": CallFirst,
    "yaochu_first": DiscardYaochuFirst,
}


def make_policy(name, seed=0):
    return POLICIES[name](seed=seed)
