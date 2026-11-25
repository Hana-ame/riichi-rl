"""Phase 0 引擎/观测/训练修复回归测试（纯 Python，不依赖 PyTorch）。

覆盖四大问题修复：
1. 副露和牌（副露断幺九全顺子 30 符、副露役牌、副露宝牌/赤五、碰优先于吃、禁响应加杠）；
2. 弃和观测（三家对手立直平面/标量、立直后全员弃牌现物平面、响应弃牌者相对座位）；
3. 流听规则（2 听 2 不听 ±1500、庄家流局听牌连庄、子家和牌轮庄、王牌 14 张、杠后听牌）；
4. 相对吃牌槽位映射与所有修改文件编译检查。

运行：python -m tests.test_phase0_fixes
"""

from __future__ import annotations

import os
import py_compile
import unittest

import tests.utils as U
from mahjong import env as E
from mahjong import hanchan as HC
from mahjong import obs2 as O2
from mahjong import mp_env as MP


class TestOpenHandFixes(unittest.TestCase):
    """副露和牌、役牌、宝牌与响应优先级回归。"""

    def test_open_tanyao_all_sequences_30fu_can_win(self):
        # 副露断幺九全顺子（吃 234m + 暗牌 345p 456p 567s 66s）-> 应为 30 符 1 番可自摸/荣和
        env = U.make_env(seed=10, dealer=0)
        p = env.players[0]
        p.melds = [("chi", 1, 3)]
        p.hand = [11, 12, 13, 12, 13, 14, 22, 23, 24, 23, 23]
        p.last_drawn = 23
        self.assertTrue(env._can_tsumo(0), "副露断幺九全顺子必须能自摸和牌")
        res = env._finish_tsumo(0)
        self.assertEqual(res.result_type, "tsumo")
        self.assertEqual(res.fu, 30, "副露 0 符必须补 2 符进位为 30 符")
        self.assertIn("断幺九", res.yaku)

    def test_open_yakuhai_dragon_and_wind(self):
        # 碰白板(31) + 123m 456p 789s 22s -> 役牌:白可和
        env = E.MahjongEnv(seed=11, dealer=0, round_wind=27)
        p = env.players[0]
        p.melds = [("pon", 31, 1)]
        p.hand = [0, 1, 2, 12, 13, 14, 24, 25, 26, 19, 19]
        p.last_drawn = 19
        self.assertTrue(env._can_tsumo(0), "碰白板必须成役牌可和")
        res = env._finish_tsumo(0)
        self.assertIn("役牌:白", res.yaku)

        # 庄家(自风东=27，场风东=27)碰东 -> 连风牌应同时计场风+自风
        env2 = E.MahjongEnv(seed=12, dealer=0, round_wind=27)
        p2 = env2.players[0]
        p2.melds = [("pon", 27, 2)]
        p2.hand = [0, 1, 2, 12, 13, 14, 24, 25, 26, 19]
        env2.last_discard = 19
        env2.last_discard_player = 1
        env2.phase = E.Phase.RESPONSE
        self.assertTrue(env2._can_ron(0), "连风东碰必须可荣和")
        res2 = env2._finish_ron(0, 1, 19)
        self.assertIn("役牌:场风", res2.yaku)
        self.assertIn("役牌:自风", res2.yaku)

    def test_open_meld_counts_dora(self):
        env = U.make_env(seed=13, dealer=0)
        env.dora_indicators = [30]  # 北 -> 宝牌东(27)
        env.dora_indicator = 30
        p = env.players[0]
        p.melds = [("pon", 27, 1)]
        p.hand = [0, 1, 2, 12, 13, 14, 24, 25, 26, 19, 19]
        self.assertEqual(env._dora_count(0, [], 19), 3, "副露中的 3 张宝牌必须计入 dora_count")

    def test_pon_beats_chi_and_no_response_addkan(self):
        env = U.make_env(seed=14, dealer=0)
        env.start_turn()
        # 玩家 0 打出 3m(2)：下家(1)有 1m2m 可吃，对家(2)有 3m3m 可碰
        env.players[1].hand = [0, 1, 9, 10, 11, 18, 19, 20, 27, 28, 29, 30, 31]
        env.players[2].hand = [2, 2, 9, 10, 11, 18, 19, 20, 27, 28, 29, 30, 31]
        # 玩家 3 有 3m 的碰，测试响应阶段不生成 ADD_KAN
        env.players[3].melds = [("pon", 2, 0)]
        env.players[3].hand = [9, 10, 11, 18, 19, 20, 27, 28, 29, 30]
        env.last_discard = 2
        env.last_discard_player = 0
        env.phase = E.Phase.RESPONSE
        env.pending = [1, 2, 3]

        acts3 = env.legal_actions(3)
        self.assertFalse(any(a.type == E.ACT_ADD_KAN for a in acts3), "他家弃牌响应阶段不能加杠")

        env.step({
            1: E.Action(E.ACT_CHI, 0, 2),
            2: E.Action(E.ACT_PON, 2),
            3: E.PASS_ACTION,
        })
        self.assertEqual(env.turn_player, 2, "碰必须优先于吃")
        self.assertEqual(len(env.players[2].melds), 1)
        self.assertEqual(len(env.players[1].melds), 0)


class TestDefenseObsFixes(unittest.TestCase):
    """对手立直与现物安全牌特征回归。"""

    def test_opponent_riichi_and_genbutsu_encoded(self):
        env = U.make_env(seed=20, dealer=0)
        env.start_turn()
        # 玩家 1（视角 0 的下家）先打 5m(4) 再立直打 9p(17)，随后玩家 2 打 1s(18)
        p1 = env.players[1]
        p1.river = [4]
        p1.genbutsu = {4}
        p1.riichi = True
        p1.genbutsu.update(p1.river)
        p1.river.append(17)
        p1.genbutsu.add(17)
        # 模拟玩家 2 弃牌 1s(18)
        env.players[2].hand = [18] + [0] * 13
        env.phase = E.Phase.DISCARD
        env.pending = [2]
        env.step({2: E.Action(E.ACT_DISCARD, 18)})

        enc = O2.ObsEncoder()
        planes, scalar = enc.encode(env, 0)
        self.assertEqual(planes.shape, (O2.N_PLANES, 34))
        self.assertEqual(scalar.shape, (O2.SCALAR_DIM,))
        # 对手 1（相对座位 1）立直平面 22 应全 1，标量 26 应为 1
        self.assertEqual(float(planes[22, 0]), 1.0)
        self.assertEqual(float(scalar[26]), 1.0)
        # 4(立直前牌河), 17(立直宣言牌), 18(立直后他家弃牌) 都必须在对手 1 现物平面 25 上为 1
        for safe_t in (4, 17, 18):
            self.assertEqual(float(planes[25, safe_t]), 1.0, f"牌 {safe_t} 应为下家现物")
        # 响应阶段弃牌者是玩家 2（相对座位 2）-> scalar[30] == 1.0
        self.assertEqual(float(scalar[30]), 1.0)


class TestRyuukyokuAndHanchanFixes(unittest.TestCase):
    """流局罚符、连庄轮庄与杠后听牌回归。"""

    def test_two_tenpai_penalty_is_1500(self):
        env = U.make_env(seed=30, dealer=0)
        # 0 和 1 听牌，2 和 3 不听
        tenpai_13 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 12, 13]
        noten_13 = [0, 2, 4, 6, 8, 9, 11, 13, 15, 17, 27, 29, 31]
        env.players[0].hand = list(tenpai_13)
        env.players[1].hand = list(tenpai_13)
        env.players[2].hand = list(noten_13)
        env.players[3].hand = list(noten_13)
        res = env._finish_ryuukyoku("exhaustive")
        self.assertEqual(res.points_delta, [1500, 1500, -1500, -1500],
                         "2 听 2 不听必须按总池 3000 结算为 ±1500")

    def test_kan_tenpai_check(self):
        env = U.make_env(seed=31, dealer=0)
        p = env.players[0]
        p.melds = [("ankan", 0, 0)]
        # 10 张暗牌：3 面子 + 单骑听东(27)
        p.hand = [9, 10, 11, 12, 13, 14, 18, 19, 20, 27]
        self.assertTrue(env._is_tenpai(0), "暗杠后 10 张暗牌听单骑必须判为听牌")

    def test_hanchan_renchan_and_rotation(self):
        class _Res:
            def __init__(self, rtype, winner=None, tenpai=None):
                self.result_type = rtype
                self.winner = winner
                self.tenpai = tenpai
                self.points_delta = [0, 0, 0, 0]
                self.kyokuten = 0

        h = HC.HanchanState(dealer=0)
        # 庄家流局听牌 -> 连庄，kyoku 不变，honba +1
        h.advance(_Res("ryuukyoku", tenpai=[True, False, False, False]))
        self.assertEqual((h.dealer, h.kyoku, h.honba), (0, 1, 1))
        # 子家(2)和牌 -> 轮庄到下家(1)，不是 2，honba 归零，kyoku 推进到 2
        h.advance(_Res("ron", winner=2))
        self.assertEqual((h.dealer, h.kyoku, h.honba), (1, 2, 0))
        # 庄家(1)和牌 -> 连庄，honba +1，kyoku 不变
        h.advance(_Res("tsumo", winner=1))
        self.assertEqual((h.dealer, h.kyoku, h.honba), (1, 2, 1))



    def test_dealer_wall_70_draws_rinshan_and_ura_alignment(self):
        env = U.make_env(seed=42)
        self.assertEqual(env._drawable_left(), 70, "0 杠活牌山必须精确为 70 张")
        ok, tot, _ = env.check_tile_conservation()
        self.assertTrue(ok and tot == 136)

        env.start_turn()  # 69 left
        env.players[0].hand = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        env._do_kan(0, E.Action(E.ACT_ANKAN, 0))
        self.assertEqual(env._drawable_left(), 68, "每次开杠活牌山仅减 1 张")
        self.assertEqual(len(env.dora_indicators), 2)
        env._set_ura_dora()
        self.assertEqual(len(env.ura_dora_indicators), 2, "里宝牌指示牌张数必须与表宝牌 1-to-1 对齐")

        env.wall_idx = env._live_wall_end()
        env.players[0].hand = [1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.assertFalse(any(a.type == E.ACT_ANKAN for a in env.legal_actions(0)),
                         "活牌山剩余 0 张时必须禁止开杠")


class TestChiSlotAndCompile(unittest.TestCase):
    def test_relative_chi_slots(self):
        # 顺子 234m(start=1)：吃 2m(1)->左吃44, 吃 3m(2)->中吃45, 吃 4m(3)->右吃46
        self.assertEqual(MP._slot_of(E.Action(E.ACT_CHI, 1, 1)), MP.SLOT_CHI_BASE + 0)
        self.assertEqual(MP._slot_of(E.Action(E.ACT_CHI, 1, 2)), MP.SLOT_CHI_BASE + 1)
        self.assertEqual(MP._slot_of(E.Action(E.ACT_CHI, 1, 3)), MP.SLOT_CHI_BASE + 2)

    def test_patched_files_compile(self):
        for rel in (
            "mahjong/hand.py",
            "mahjong/scoring.py",
            "mahjong/refengine.py",
            "mahjong/env.py",
            "mahjong/hanchan.py",
            "mahjong/obs2.py",
            "mahjong/mp_env.py",
            "train_hanchan.py",
        ):
            py_compile.compile(os.path.join(U.ROOT, rel), doraise=True)


if __name__ == "__main__":
    unittest.main()
