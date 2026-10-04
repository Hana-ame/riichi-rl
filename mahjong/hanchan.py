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

    def __init__(self, dealer=0, start_points=25000):
        self.round = "E"
        self.kyoku = 1
        self.dealer = dealer
        self.honba = 0
        self.sticks = [start_points] * 4
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
        )

    def advance(self, result):
        """结算一局后推进场次。

        result: MahjongEnv.GameResult

        Returns:
            (next_dealer, delta_applied)
        """
        delta = result.points_delta
        for i in range(4):
            self.sticks[i] += delta[i]

        if result.result_type == "ryuukyoku":
            # 流局：连庄棒 +1，庄家不变，但**场次必须推进**，
            # 否则半庄永远打不完（kyoku 不变 -> while 死循环）。
            self.honba += 1
            self._advance_round()
            return self.dealer, delta

        # 和了：和了者成为庄家，棒数清零
        self.dealer = result.winner
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
