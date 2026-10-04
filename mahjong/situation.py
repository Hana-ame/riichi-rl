"""局势评估：从**规则可见信息**判断当前处于什么局面。

⚠️ 与「牌效」的边界（铁律）
────────────────────────────────────────────────────────────
本模块只回答「局面处在哪个阶段」：
  - 分差多少（点数是公开计分器）
  - 场次进度（东几局、连庄数）
  - 自己离听牌多远（向听数是规则定义）
  - 有没有人立直（立直是公开宣告）
  - 已经巡了多少巡

本模块**不回答**「该打哪张」「哪张牌安全」：
  - 不含任何牌与牌的相关性
  - 不含进张数、危险牌、期望值
  - 输出的 situation 标签只是**局势分类**，
    供策略网络（而非人类规则）决定放弃还是继续

换句话说：这是「牌局的比分板」，不是「麻将技巧」。
────────────────────────────────────────────────────────────
"""

from __future__ import annotations

# 局势分类
SIT_NORMAL = "normal"          # 均势，正常打
SIT_AHEAD = "ahead"            # 大幅领先，可以稳
SIT_BEHIND = "behind"          # 落后，需要搏
SIT_LAST_CHANCE = "last_chance"  # 末局大幅落后，必须搏
SIT_DEFENSE = "defense"        # 有人立直，必须防
SIT_DESERVE = "deserve"        # 已无希望（弃听条件）


class Situation:
    __slots__ = ("label", "point_gap", "is_last_round", "riichi_count",
                 "turn_ratio", "my_shanten", "honba", "round_label")

    def __init__(self, label, point_gap, is_last_round, riichi_count,
                 turn_ratio, my_shanten, honba, round_label):
        self.label = label
        self.point_gap = point_gap
        self.is_last_round = is_last_round
        self.riichi_count = riichi_count
        self.turn_ratio = turn_ratio
        self.my_shanten = my_shanten
        self.honba = honba
        self.round_label = round_label

    def __repr__(self):
        return (f"Situation[{self.label} gap={self.point_gap:+.0f} "
                f"riichi={self.riichi_count} shanten={self.my_shanten} "
                f"{self.round_label}]")

    def to_vec(self):
        """给观测网络用的局势特征（全部是规则可见信息）。"""
        import numpy as np
        labels = [SIT_NORMAL, SIT_AHEAD, SIT_BEHIND, SIT_LAST_CHANCE,
                  SIT_DEFENSE, SIT_DESERVE]
        onehot = [1.0 if self.label == L else 0.0 for L in labels]
        return np.array(
            onehot + [
                max(-1.0, min(1.0, self.point_gap / 25000.0)),
                1.0 if self.is_last_round else 0.0,
                self.riichi_count / 4.0,
                self.turn_ratio,
                min(1.0, max(0.0, self.my_shanten / 8.0)),
                min(1.0, self.honba / 4.0),
            ], dtype=np.float32)


def assess(env, hstate, p_idx, my_shanten=8):
    """评估 p_idx 所处的局面。

    参数:
        env: MahjongEnv（当前局）
        hstate: HanchanState（半庄进度）
        p_idx: 评估对象
        my_shanten: 该玩家的向听数（由调用方传入，避免重复计算）

    返回 Situation。
    """
    sticks = hstate.sticks
    my_pts = sticks[p_idx]

    # 分差：相对场上的领先/落后
    others = [sticks[i] for i in range(4) if i != p_idx]
    best_other = max(others)
    worst_other = min(others)
    point_gap = my_pts - best_other        # 领先最多的对手
    gap_to_worst = my_pts - worst_other

    # 末局判定：南 4 且点数已分出差距
    is_last_round = (hstate.round == "S" and hstate.kyoku >= 4)

    # 立直家数（公开宣告）
    riichi_count = sum(1 for q in env.players if q.riichi)

    # 巡目进度
    turn_ratio = min(1.0, env.turn / 72.0)

    round_label = f"{hstate.round}{hstate.kyoku}"

    # ---- 分类 ----
    # 优先级：**末局搏命 > 有人立直（需防守） > 大幅领先/落后 > 均势**
    #
    # 修正记录：原先「大幅领先/落后」排在「有人立直」之前，
    # 导致自然对局里 defense 从不出现（behind/ahead 先匹配掉了）——
    # 实测 60 局里 defense=0、last_chance=0。
    # 现在把「是否需要防守/是否末局」放在更前面，因为它们是**动作约束**
    # （要不要全力防、要不要搏命），比分差标签更直接地决定行为。
    START = 25000
    if is_last_round and point_gap <= -12000:
        label = SIT_LAST_CHANCE        # 末局搏命：最高优先
    elif riichi_count >= 1:
        label = SIT_DEFENSE            # 有人立直：必须防
    elif point_gap >= 12000:
        label = SIT_AHEAD              # 大幅领先：可以稳
    elif point_gap <= -12000:
        label = SIT_BEHIND             # 大幅落后：要搏
    else:
        label = SIT_NORMAL

    return Situation(
        label=label,
        point_gap=float(point_gap),
        is_last_round=is_last_round,
        riichi_count=riichi_count,
        turn_ratio=float(turn_ratio),
        my_shanten=int(my_shanten),
        honba=hstate.honba,
        round_label=round_label,
    )


def deserves_give_up(sit, wall_left_ratio=0.34):
    """「弃听」判定：是否应该放弃和了、保命为主。

    条件（全部是规则信息）：
      - 分差已经追不回来（大幅落后）
      - 且不是末局搏命局
      - 且已经过了大半（牌墙剩余不多）
      - 且自己还没听牌

    这不是「人类技巧」，是**局面算术**：
    落后 X 点 / 剩余 Y 张牌，就算每张都最优也追不回，那就是数学结论。
    """
    START = 25000
    max_win_per_deal = 12000          # 役满级别的上限
    max_deals_left = max(1, int(wall_left_ratio * 4))
    reachable = max_win_per_deal * max_deals_left

    too_far_behind = (-sit.point_gap) > reachable
    too_late = sit.turn_ratio > 0.66
    not_tenpai = sit.my_shanten > 0
    not_last_chance_bet = not (
        sit.is_last_round and sit.point_gap <= -12000)

    return (too_far_behind and too_late and not_tenpai
            and not_last_chance_bet)
