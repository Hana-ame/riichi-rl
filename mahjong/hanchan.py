"""半庄（hanchan）场次管理：东场→南场、连庄、点数结算。

对应 RULES.md §3.1、§13：半庄 = 东 1-4 四局，每局可连庄。
每局结束（和了或流局）后决定下一局的庄家与场次。

这是**规则层**，不含任何策略判断。
"""

from __future__ import annotations

from mahjong import env as E
from mahjong import ruleset as RS
from mahjong.tiles import EAST, SOUTH

#: 局数上限：东 1,2,3,4 = 4 局；南 1,2,3,4 = 4 局
KYOKU_LIMIT = {"E": 4, "S": 4}


class HanchanState:
    """半庄进度。

    Attributes:
        round: "E" 或 "S"
        kyoku: 1..4（该场的第几局）
        dealer: 当前庄家座位
        honba: 连庄棒数
        sticks: 各家累积点数
        finished: 半庄是否已结束
    """

    def __init__(self, dealer=0, start_points=25000, east_only=False):
        """east_only=True 时只打东 1–4（东风战），东 4 结束即终局。"""
        self.east_only = bool(east_only)
        self.round = "E"
        self.kyoku = 1
        self.dealer = dealer
        self.honba = 0
        self.sticks = [start_points] * 4
        self.kyokuten = 0          # 供托（立直棒），流局保留到下一局
        self.finished = False

    def wind_of(self, player_idx):
        """该座位在本场的场风 tile id。"""
        base = EAST if self.round == "E" else SOUTH
        offset = (player_idx - self.dealer) % 4
        return base + offset

    def seat_wind_of(self, player_idx):
        """该座位的自风（相对庄家，跨场不变）。"""
        return 27 + ((player_idx - self.dealer) % 4)

    def current_index(self):
        base = 0 if self.round == "E" else 4
        return base + self.kyoku - 1

    def progress_label(self):
        return f"{self.round}{self.kyoku} ({self.honba} 连庄)"

    def new_env(self, seed=None):
        """按当前场次开一局。"""
        return E.MahjongEnv(
            seed=seed,
            dealer=self.dealer,
            honba=self.honba,
            round_wind=self.wind_of(self.dealer),
            kyoku=self.kyoku,
            start_kyokuten=self.kyokuten,
            start_points=self.sticks,
        )

    def advance(self, result):
        """结算一局后推进场次。

        点数接力用 `result.scores`（绝对点数）而不是 `points_delta`：
        delta 里不含立直时直接扣走的 1000 点（供托），只加 delta 会让
        「sticks 总和 + 供托」不再等于 100,000。
        对没有 `scores` 的旧式/测试 result（只有 points_delta）保留
        delta 回退路径。

        result: MahjongEnv.GameResult

        Returns:
            (next_dealer, delta_applied)
        """
        delta = list(result.points_delta)
        scores = getattr(result, "scores", None)
        if scores is not None:
            self.sticks = [int(x) for x in scores]
        else:
            for i in range(4):
                self.sticks[i] += delta[i]
        # 供托接力：和牌时已由 env 结算给和牌者（result.kyokuten=0）；
        # 流局时保留，带到下一局。
        self.kyokuten = getattr(result, "kyokuten", 0)

        if result.result_type == "ryuukyoku":
            # 流局：本场 +1；庄家听牌则连庄（不推进局数），不听则轮庄并推进局数。
            # 兼容无 tenpai 字段的旧测试 stub（视为庄家不听）。
            self.honba += 1
            tenpai = getattr(result, "tenpai", None)
            dealer_tenpai = bool(tenpai and 0 <= self.dealer < len(tenpai) and tenpai[self.dealer])
            if dealer_tenpai and self.honba <= 8:
                return self.dealer, delta
            self.dealer = (self.dealer + 1) % 4
            self._advance_round()
            return self.dealer, delta

        # 和了：庄家和牌连庄（本场 +1，局数不变）；子家和牌轮庄（下家坐庄，本场清零，局数推进）
        if result.winner == self.dealer and self.honba < 8:
            self.honba += 1
            return self.dealer, delta
        self.dealer = (self.dealer + 1) % 4
        self.honba = 0
        self._advance_round()
        return self.dealer, delta

    def _advance_round(self):
        """推进场次：东1→东2→…→东4→南1→…→结束。

        流局与和了都要调用，否则流局会让半庄无限延续。
        """
        if self.finished:
            return          # 已结束，不要再推进
        if self.kyoku >= KYOKU_LIMIT[self.round]:
            if self.round == "E":
                if self.east_only:
                    self.finished = True      # 东风战：东 4 打完即结束
                    return
                self.round = "S"
                self.kyoku = 1
            else:
                self.finished = True
        else:
            self.kyoku += 1

    def final_placement(self):
        """终局位次（点数降序，同点按座次顺位）。"""
        order = sorted(range(4), key=lambda i: (-self.sticks[i], i))
        return order

    def to_dict(self):
        return {
            "round": self.round,
            "kyoku": self.kyoku,
            "dealer": self.dealer,
            "honba": self.honba,
            "sticks": list(self.sticks),
            "east_only": self.east_only,
            "finished": self.finished,
        }


def placement_bonus(order, hanchan=True):
    """终局顺位点。半庄 1st/2nd/3rd/4th = 30000/10000/-10000/-30000。"""
    if hanchan:
        table = [30000, 10000, -10000, -30000]
    else:
        table = [15000, 5000, -5000, -15000]
    return {p: table[r] for r, p in enumerate(order)}


def point_gap(sticks):
    """各家与庄家的分差（正值 = 领先）。"""
    dealer_pts = sticks[0]  # 仅当 dealer==0 时有意义
    return [s - dealer_pts for s in sticks]
