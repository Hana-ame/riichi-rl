"""以和牌为终点的纯组合非 ML 模块：向听 + 进张(ukeire) + 摸牌概率。

不做任何训练/学习——全部是确定性计算：
  * 向听：REF.ref_shanten（开源向听数，和了=-1，听牌=0）
  * 进张(ukeire)：对当前手牌，枚举 34 种牌，数「摸进后向听严格下降」的牌种数
  * 摸牌概率：按可见牌(手牌+副露+自己牌河)扣除后，剩余每种牌的枚数 / 总剩余张数

和牌作为 destination：向听=0 且能自摸/荣和即抵达；否则用摸牌概率估计下一步
向听下降的期望，作为「离胡牌多远」的可解释量度。

用法（独立函数，不依赖训练）：
    from shanten_odds import analyze, draw_probability
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mahjong import refengine as REF
from mahjong.tiles import NUM_TILES

TOTAL_COPIES = 4


def _safe_shanten(counts):
    try:
        return REF.ref_shanten(counts)
    except Exception:
        return 8          # 非法手牌形状按最远处理


def ukeire(counts, meld_count=0):
    """进张：摸进哪几种牌能让向听严格下降。

    返回 (uke_tile_types, shanten)。uke_tile_types 是 [(tile34, 之后向听)] 列表。
    与 mahjong/policies.py 的牌效教师 tie-break 同一口径（标准牌效）。
    """
    sh = _safe_shanten(counts)
    if sh < 0:                     # 已和牌
        return [], sh
    uke = []
    for t in range(NUM_TILES):
        if counts[t] >= TOTAL_COPIES:
            continue
        counts[t] += 1
        ns = _safe_shanten(counts)
        counts[t] -= 1
        if ns < sh:
            uke.append((t, ns))
    return uke, sh


def visible_counts(env, p_idx):
    """全局可见牌：所有人的手牌+副露+牌河（简化：本模块只扣自己的手牌+副露+牌河）。

    真实场景对手手牌不可见，故只扣确定可见的部分。返回 34 长度数组。
    """
    c = [0] * NUM_TILES
    me = env.players[p_idx]
    for t in me.river:
        c[t] += 1
    for kind, t, _ in me.melds:
        if kind == "chi":
            for tt in (t, t + 1, t + 2):
                c[tt] += 1
        elif kind in ("pon", "minkan", "ankan", "addkan"):
            c[t] += 4 if kind != "pon" else 3
    return c


def draw_probability(env, p_idx, counts=None):
    """摸到每种牌的概率（按剩余不可见牌计算）。

    返回 list[(tile34, prob)]，prob = 剩余枚数 / 总剩余张数，和为 1。
    """
    if counts is None:
        counts = env.players[p_idx].full_counts()
    vis = visible_counts(env, p_idx)
    remaining = [max(0, TOTAL_COPIES - counts[t] - vis[t]) for t in range(NUM_TILES)]
    total = sum(remaining)
    if total <= 0:
        return [(t, 0.0) for t in range(NUM_TILES)]
    return [(t, remaining[t] / total) for t in range(NUM_TILES)]


def analyze(env, p_idx):
    """一次性给出：当前向听、进张列表、进张概率、摸到任一进张的概率。

    返回 dict：
      shanten      当前向听数（-1=已和）
      uke          [(tile34, 之后向听, 摸到它的概率)]
      uke_tiles    进张牌种数（标准牌效的 ukeire）
      p_advance    下一巡摸到任一进张的概率
      tenpai       是否听牌（向听=0）
    """
    me = env.players[p_idx]
    counts = me.full_counts()
    uke, sh = ukeire(counts)
    prob = dict(draw_probability(env, p_idx, counts))
    uke_rows = [(t, ns, prob.get(t, 0.0)) for t, ns in uke]
    p_adv = sum(pr for _, _, pr in uke_rows)
    return {
        "shanten": sh,
        "uke": uke_rows,
        "uke_tiles": len(uke_rows),
        "p_advance": p_adv,
        "tenpai": sh == 0,
    }


def expected_turns_to_tenpai(env, p_idx, max_turns=18):
    """期望几巡到听牌/和牌（非 ML：按剩余枚数做前向概率传播）。

    状态抽象：只跟踪「向听等级」概率分布 dist[shanten]。每巡对每个仍 >0 的向听
    等级估一个进张概率 p_adv，然后 进张(概率 p_adv)->向听-1，未进张->保持。

    **已知近似（非精确解）**：p_adv 用开局 counts 的进张集合，不随摸打更新手牌。
    有意简化，换取可解释的期望巡数，避免 (counts,turn) 状态数指数级爆炸。
    """
    me = env.players[p_idx]
    counts0 = list(me.full_counts())
    # 缓存：p_adv 在近似下只依赖 counts0，算一次即可
    uke0, _ = ukeire(counts0)
    uke_tiles0 = {t for t, _ in uke0}
    prob0 = dict(draw_probability(env, p_idx, counts0))
    p_adv0 = sum(prob0.get(t, 0.0) for t in uke_tiles0)

    dist = {_safe_shanten(counts0): 1.0}
    tenpai_turn = None
    win_turn = None
    for turn in range(1, max_turns + 1):
        new_dist = {}
        for sh, mass in dist.items():
            if sh <= 0:
                new_dist[sh] = new_dist.get(sh, 0.0) + mass
                continue
            if p_adv0 <= 0:
                new_dist[sh] = new_dist.get(sh, 0.0) + mass
                continue
            # 进张 -> 向听 -1；未进张 -> 保持
            new_dist[sh - 1] = new_dist.get(sh - 1, 0.0) + mass * p_adv0
            new_dist[sh] = new_dist.get(sh, 0.0) + mass * (1 - p_adv0)
        dist = new_dist
        if tenpai_turn is None:
            acc = sum(m for sh, m in dist.items() if sh == 0)
            if acc > 0.5:
                tenpai_turn = turn
        win_acc = sum(m for sh, m in dist.items() if sh < 0)
        if win_acc > 0.5 and win_turn is None:
            win_turn = turn
            break
    return {
        "tenpai_turns": tenpai_turn,      # None = max_turns 内到不了听牌(>50%)
        "win_turns": win_turn,
        "final_dist": dist,               # 巡末向听分布
    }


if __name__ == "__main__":
    # 冒烟：起一局看四家向听与进张概率
    from mahjong.env import MahjongEnv
    env = MahjongEnv(seed=8801)
    for p in range(4):
        a = analyze(env, p)
        t = expected_turns_to_tenpai(env, p, max_turns=18)
        print(f"席{p} 向听={a['shanten']} 进张种数={a['uke_tiles']} "
              f"摸到进张概率={a['p_advance']:.3f} "
              f"到听牌期望巡={t['tenpai_turns']} 到和牌期望巡={t['win_turns']}")
