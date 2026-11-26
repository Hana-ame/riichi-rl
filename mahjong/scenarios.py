"""策略奖励与观测增强：弃听 / 全防 / 场次 / 分差。

这是**奖励与观测层**，不是规则层。规则层（hand/scoring/ruleset）已对照
MahjongRepository/mahjong 2.0.0 核验，本文件不改任何规则。

四件事
------
1. 弃听场景：局势已不可挽回时，正确动作是**保命**（少放铳）而不是搏和。
   奖励上体现为：落后局放铳的惩罚更重，和了反而不一定赚。
2. 全防场景：有人立直时，放铳=直接把点数送给对手。
   奖励上体现为：立直家存在时放铳的惩罚显著加大。
3. 东场/南场：末局（南4）大幅落后时，从「保命」切到「搏命」。
4. 分差：所有奖励都以「相对领先多少」为核心，而不是绝对点数。

奖励公式（训练用）
------------------
    r = w_point  * (相对分差变化 / 10000)
      + w_shape * (保持手牌完整，即不增加向听)
      - w_deal  * 放铳惩罚（按局势放大）
      + w_surv  * 安全局奖励（弃听/防守局里不输钱）
"""

from __future__ import annotations

from mahjong import situation as SIT
from mahjong.situation import Situation, assess, deserves_give_up


# 奖励权重（默认，可在训练脚本里覆盖）
# 注：r_point = delta / 10000 已经完整包含放铳失分与和牌得分；
# 过大的额外放铳惩罚(w_deal)与弃听保命奖(w_survive=0.8)会把 RL 推向极度消极弃和，
# 因此默认归零，仅在显式调参时启用轻度塑形。
W_POINT = 1.0        # 真实点数变化（主目标）
W_SHAPE = 0.0        # 向听塑形（从 BC 起步建议为 0）
W_DEAL_BASE = 0.0    # 额外放铳惩罚（默认 0，避免与 r_point 双重扣分）
W_DEAL_DEFENSE = 1.5 # 有人立直时的放铳惩罚倍数
W_DEAL_LAST = 1.5    # 末局搏命时降低额外放铳惩罚
W_SURVIVE = 0.0      # 弃听局保命奖（默认 0，避免奖励消极弃和）


class ScenarioFlags:
    """一局内某玩家的场景标记（供奖励与观测使用）。"""

    __slots__ = ("situation", "give_up", "defense_mode")

    def __init__(self, situation, give_up, defense_mode):
        self.situation = situation      # Situation 对象
        self.give_up = give_up          # 是否进入弃听
        self.defense_mode = defense_mode  # 是否进入全防


def reward_for_result(player, result, sit_before, hstate_before=None,
                      give_up_flag=False,
                      w_point=W_POINT, w_shape=0.0,
                      w_deal=W_DEAL_BASE, w_survive=0.0, w_place=0.0):
    """终局奖励：这一局的结果 + 当时的局势。

    参数:
        player: 玩家 idx
        result: GameResult
        sit_before: Situation 替身（只需 riichi_count / is_last_round / point_gap）
        give_up_flag: 该局是否处于弃听场景
        w_place: 顺位 pt 奖励权重（天凤：1位+30/2位+10/3位-10/4位-30）。
            >0 时按最终点数排名次加顺位 pt，使 RL 目标对齐天凤排位
            （名次主导，点数只是修正项），而不是只优化绝对点数。
    """
    delta = result.points_delta[player]
    r_point = w_point * (delta / 10000.0)

    # 1.5 顺位 pt（天凤排位目标）：按最终点数排名次
    #   量级：原始顺位 pt(±30) 远大于点数 reward(±2)，会让 critic 目标方差爆炸
    #   （vloss 从 0.5 涨到 11+）。归一化 /10 -> ±3/±1，顺位仍主导但 value 可学。
    r_place = 0.0
    if w_place > 0:
        scores = getattr(result, "scores", None)
        if scores is None:
            scores = [result.points_delta[i] for i in range(4)]
        my = scores[player]
        rank = sum(1 for s in scores if s > my)  # 0=1位
        r_place = w_place * (30.0, 10.0, -10.0, -30.0)[min(rank, 3)] / 10.0

    # 2. 放铳惩罚
    r_deal = 0.0
    dealt_in = (result.from_player == player)
    if dealt_in:
        mult = w_deal
        if sit_before.riichi_count >= 1:
            mult *= W_DEAL_DEFENSE      # 有人立直，放铳代价更大
        if sit_before.is_last_round and sit_before.point_gap <= -12000:
            mult *= (1.0 / W_DEAL_LAST)  # 搏命局放铳无所谓
        r_deal = -mult * (abs(delta) / 10000.0)

    # 3. 安全局奖励（弃听局里不输）
    r_surv = 0.0
    if w_survive > 0 and give_up_flag and delta >= 0:
            r_surv = w_survive

    return r_point + r_place + r_deal + r_surv + w_shape * 0.0


def situation_features(env, hstate, p_idx, my_shanten):
    """给观测网络追加的局势特征（12 维，全部规则可见信息）。"""
    sit = assess(env, hstate, p_idx, my_shanten)
    give_up = deserves_give_up(sit)
    return sit.to_vec(), sit, give_up


def augment_obs(planes, scalar, sit_vec):
    """把局势特征接到标量段后面。"""
    import numpy as np
    return np.concatenate([scalar, sit_vec]).astype(np.float32)
