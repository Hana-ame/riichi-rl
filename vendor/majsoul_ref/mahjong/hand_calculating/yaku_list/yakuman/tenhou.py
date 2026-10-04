from collections.abc import Collection, Sequence

from majsoul_ref.mahjong.hand_calculating.yaku import Yaku


class Tenhou(Yaku):
    """天和: Dealer wins on the initial draw."""

    yaku_id = 115
    name = "Tenhou"
    han_closed = 13
    is_yakuman = True

    def is_condition_met(self, hand: Collection[Sequence[int]], *args) -> bool:
        """Return True unconditionally; this yaku is awarded by the hand evaluation logic."""
        return True
