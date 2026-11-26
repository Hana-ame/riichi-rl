"""dest_prob_v3 参数变体排位：def15/def20/nocall vs dest_prob_v3/pure_shanten/eff_wuke_d2。

用法：
    python arena_dest_prob_param.py --games 400 --out rules/ELO_dest_prob_param.md
"""
from __future__ import annotations
import os, sys, argparse
sys.path.insert(0, os.getcwd())

from mahjong import policies as POL
import dest_prob_v3 as _DP
_DP.register_policy("dest_prob_v3")
POL.POLICIES["dest_prob_v3_def15"] = lambda seed=0: _DP.DestProbV3(seed=seed, defense_weight=1.5)
POL.POLICIES["dest_prob_v3_def20"] = lambda seed=0: _DP.DestProbV3(seed=seed, defense_weight=2.0)
POL.POLICIES["dest_prob_v3_nocall"] = lambda seed=0: _DP.DestProbV3(seed=seed, enable_calls=False)

import arena_elo as AE
from arena_elo import RuleAgent

_orig_build = AE.build_participants
def _build_with_params(extra_ckpts=None):
    agents = _orig_build(extra_ckpts)
    for nm, label in [("dest_prob_v3_def15", "dest_prob_v3_def15(高防1.5)"),
                      ("dest_prob_v3_def20", "dest_prob_v3_def20(高防2.0)"),
                      ("dest_prob_v3_nocall", "dest_prob_v3_nocall(禁副露)")]:
        agents.append(RuleAgent(nm, label))
    return agents
AE.build_participants = _build_with_params

if __name__ == "__main__":
    AE.main()
