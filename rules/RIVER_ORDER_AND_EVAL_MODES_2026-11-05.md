# 两项补充核验：东风/半庄评估入口、hand_predictor 是否含牌河顺序/向听输出

## 1. 东风 + 半庄分数评估：已支持（`eval_hanchan.py` 双模式）

```python
# eval_hanchan.py
151: ap.add_argument("--mode", default="hanchan", choices=["hanchan", "east"],
120: def run(fn, n_hanchans, label, east_only=False):
125:     h = HanchanState(dealer=i % 4, east_only=east_only)
```

- 参数名：`--mode`；取值 `hanchan`（默认，E1→S4 半庄）/ `east`（E1–E4 东风战）；
- 默认值：**hanchan**；
- 实测：teacher47_100h_v2 半庄 80 局 局和牌 33.75%（座位≈8.44%）、
  东风 40 局 局和牌 22.5%（座位≈5.62%）。
- 其他评估脚本：`eval_model.py` / `eval_student.py` **没有** `--mode` 参数
  （仍是半庄/默认流程）。若要求「所有 eval 脚本统一双模式」，还差这一处小改；
  作为「东风+半庄都作为评估对象」的要求，`eval_hanchan.py` 已满足。

## 2. 牌河顺序→手牌形状/向听：**仍未完成**（明确列为剩余项）

`hand_predictor.py` 实测特征/输出：

- **输入特征**（`build_feature`）：
  `hand_counts`(自家手牌) + `melds` + `riichi` + **`shanten`（自家向听，规则算好
  后作为输入）** + `left`(墙余) + `dora`；
  **完全没有 river / 对手弃牌 / 顺序字段**（grep `river|discard|order` 无匹配）。
- **输出头**：`win_prob`（和牌概率）、`exp_points`（预期点）、
  `turns_to_tenpai`（还几巡能听牌）；**没有 shanten 输出、没有形状输出**。
  `turns_to_tenpai` 是「未来听牌时间」的回归，不是「当前手牌形状/向听」预测。

结论：
- 「通过牌河顺序判断手牌形状/向听」= **未实现**（剩余 1 项）；
- 已有的相近能力：`opponent_model.py` 用**对手牌河 counts**（不是顺序）预测
  对手每张牌持有概率（重训 top5 0.368）；`env._is_tenpai` 可精确判**自家**听牌
  （规则信息，不是模型预测）；`hand_predictor` 输出「还几巡听牌」代理量。
- 要补：新 obs 版本加牌河**顺序/位置**平面（旧 ckpt 不兼容，需重训）+ 显式
  「对手向听/形状」预测头（标签可从 replay 数据上帝视角构造）。

## 3. 当前所有评估对象一览

| 评估对象 | 入口 | 默认 |
|---|---|---|
| 半庄分数 | `eval_hanchan.py`（默认） / `arena_elo.py` 单局 | hanchan |
| 东风战分数 | `eval_hanchan.py --mode east` | hanchan（需显式指定 east） |
| 版本晋级 | `version_ladder.py` | 固定版本池、240/500 局 |
| Elo/选员 | `arena_elo.py` | 600 局 + 前一半+随机 3 替换 |
| 听牌/放铳 | `tenpai_audit.py` | 80 局/Agent |
| 规则读数 | `engine_stats.py` | 200 局 4×greedy |
