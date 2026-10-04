from collections.abc import Collection, Sequence

from majsoul_ref.mahjong.constants import CHUN
from majsoul_ref.mahjong.hand_calculating.yaku import Yaku
from majsoul_ref.mahjong.utils import has_pon_or_kan_of


class Chun(Yaku):
    """中: Pon of red dragons."""

    yaku_id = 17
    name = "Yakuhai (chun)"
    han_open = 1
    han_closed = 1

    def is_condition_met(self, hand: Collection[Sequence[int]], *args) -> bool:
        """Check whether the hand contains a pon or kan of red dragons."""
        return has_pon_or_kan_of(hand, CHUN)
