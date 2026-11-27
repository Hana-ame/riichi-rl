# 能力验收复核（2026-11-28）：逐项证据与口径澄清

> 回应审阅清单。以下全部为文件/命令实证，不引用会话上下文猜测。

## 一、逐项核对

| # | 项目 | 状态 | 实现位置 | 实测证据 |
|---|---|---|---|---|
| 1 | 牌河顺序 → 手牌形状/向听推断 | **已实现，增益未达标** | `opp_shanten.py`（108 维基础 + 816 维顺序特征：四家各最近 6 张弃牌的位次 one-hot）；`opp_shanten_gru.py`（GRU：四家各 24 张弃牌序列 → 向听分类/听牌二分类，含无顺序 MLP 对照）；`opponent_model.py`（可见信息 → 对手每张牌持有概率） | `rules/OPP_SHANTEN_2026-11-05.md`：60 局 4×greedy 自对弈 = **22,827 样本**；顺序特征 MAE 仅降 **0.018**（验收线 0.2）→ **增益不达标** |
| 2 | 危险张判断（模型预测，非实测统计） | **已实现并验证 PASS** | `danger_model.py`（从可见信息预测「对手和牌张」概率，上帝视角标签监督）；`danger_model_agent.py`（弃牌 = logits − λ·危险分，接入主策略）；`danger_validate.py` | `rules/DANGER_VALIDATION_10000_2026-11-05.md`：10000 局/版同局占座；hybridd→hybrid **放铳 −0.88pt，95%CI [−1.57, −0.19]，PASS**；另见 `DANGER_INTEGRATION_2026-11-05.md`、`DANGER_H2H_2026-11-06.md` |
| 3 | 和牌率判断（模型预测和了概率） | **已实现（预测头已训练）** | `hand_predictor.py`：共享主干 + 3 头（**win_prob** / exp_points / turns_to_tenpai）；`checkpoints/hand_predictor.pt`（176K 已落盘） | `rules/PREDICTION_HEADS_2026-11-05.md`。另实测统计：`engine_stats.py`、`arena.py:report()`、`arena_elo.py`（每参赛者和牌率/放铳率/立直率/流局率）|
| 4 | 副露策略（吃/碰/杠/dora 完整） | **有实现，全部未达标** | `call_strategy.py`（独立叫牌小 MLP：吃/碰/明杠/暗杠/加杠 + 叫/过二分类）；`call_head.py`、`hybrid_call_agent.py`、`post_call_agent.py`；`dora_policy.py` + `dora_agent.py`（宝牌/赤牌保留加权） | `CALL_STRATEGY_2026-11-05.md`：**当前副露策略不能提升和牌率，不达标**；`CALL_STRATEGY_LM_2026-11-06.md`：**Tenhou BC 叫牌在本引擎下负收益**，需改动作空间表示；`GATE_ROUND2_2026-11-06.md`：**副露模型化/宝牌专项均未达成**；独立叫牌基线见 `ELO_with_indep_meld.md`（pon_only/kan_only/chi_only）、`ELO_with_dora.md`（eff_wuke_d2_dora） |
| 5 | 外部模型对战记录 | **有（60 局，单一模型）** | `battle_external.py`（1 席外部 vs 3 席自研）；`external_mahjonglm*.py`、`external_riichippo.py`、`external_suphx.py`、`external_elo.py`、`distill_external.py` | `rules/EXTERNAL_BATTLE_2026-11-24.md`：eff_wuke_d2 3 席 vs majonglm-100m 1 席 **60 局**，自研每席和牌 **13.9%** vs 外部 **11.7%**；外部放铳 10.0%。另 `EXTERNAL_MATCH_RECORD_2026-11-05.md`、`EXTERNAL_ELO_2026-11-05.md` 为外部模型自对局统计 |

## 二、Elo 组队与东风/半庄双局式

| 项 | 状态 | 位置 / 实测 |
|---|---|---|
| 「表现较好的一半 + 随机几个替换」组队 | **已实现** | `arena_elo.py:483-500` `reselect()`：`keep_n = len(agents)//2`，按 Elo 取前一半，`rng.sample(rest, random_fill)` 随机补几个；参数 `--warmup-games 100 / --reselect-every 100 / --random-fill 3` |
| 东风 + 半庄双局式评分 | **已实现并有实测** | `mode_elo.py`：session 级 Elo，半庄 E1–S4 / 东风 E1–E4，4 模型固定同桌、座位每场轮换。`rules/MODE_ELO_2026-11-05.md`：半庄 **teacher47_100h_v2 1548** 第一、bc_v1_last 1520；东风 **bc_v1_last 1559** 第一、greedy 1522、100h_v2 1486 |
| 说明 | `arena_elo.py` 是**单局** Elo（`play_one` 打一手即结算），不含场次；场次级由 `mode_elo.py` 补 | `mode_elo.py:1-8` docstring 明示 |

## 三、数据口径确认（重要，避免误判）

**结论：两套统计的分母是同一类「每座位每手」，差异来自参赛阵容，不是口径错误。**

| 口径 | 实现 | 分母 | 实测（teacher47_eff_wuke_d2） |
|---|---|---|---|
| 头对头 H2H / arena | `arena_elo.py:304 play_one()` = **单局（每一轮/每手）**；H2H 里 `agari[bot]/n_done`、`deal_in[bot]/n_done` | **手数**（每座位每手） | 和牌 **29–31%**、放铳 **8.3%** |
| 4 席自对弈硬门槛 | `eval_selfplay_gate.py`；`--deals` = **半庄场数**；验收分母 `N = 4 × total_hands`（座位·手，`eval_selfplay_gate.py:214-219`） | **座位·手** | 和牌 **16.5%**、放铳 **12.6%**（`rules/SELFPLAY_GATE_teacher47_eff_wuke_d2.json`） |

**为什么数值不同（同分母口径下）**：

- **门槛是 4 个同模型自对弈**（对称局，每席理论上限 25%）→ 每席和牌 16.5%、放铳 12.6%
- **H2H 是「主模型 + 1 个强对手 + 2 个弱 filler（damaten/chi_only）」** → 主模型对弱 filler 收割，和牌率被抬到 29–31%；弱 filler 攻不动它，放铳被压到 8.3%
- H2H 单表四人合计和牌 ≈74.8%，其余为流局，逻辑自洽

**因此**：29–31% / 8.3% **不违反** 15–25% / 9–20% 门槛，只是**不同对手池下的读数**；拿 H2H 数字直接比门槛会误判。

## 四、仍然真实存在的缺口（诚实清单）

1. **牌河顺序未进主策略观测**：`mahjong/obs2.py` 的牌河平面是「每种牌出现几次」的计数 + 仅 1 张 `last_discard` one-hot；顺序只在独立预测模型（opp_shanten/GRU）里用过。要用需新 obs 版本，旧 ckpt 不兼容。
2. **win_prob / 危险分未作为主策略输入通道**：`PREDICTION_HEADS_2026-11-05.md` 明示「本轮未做」；危险分目前是推理期外挂打分（`danger_model_agent.py`），不是网络输入。
3. **副露/宝牌无达标策略**：实现齐全但实测负收益（见 §一.4）。
4. **外部对战覆盖不足**：仅 60 局、1 个外部模型（mahjonglm-100m）、单一 3v1 座位配置，无跨引擎 Elo 表。
5. **东风/半庄双局式 Elo 阵容旧**：只跑过 bc_v1/greedy/100h_v2/v3 四个旧模型，未覆盖 eff_wuke_d2、dest_prob_v3、kl01 高KL 等当前主力。

## 五、本轮之前的结论修正（与本清单相关）

- PPO 高 KL 曲线与「崩溃」结论已在 `PPO_DEGRADATION_ANALYSIS.md` 修正：旧崩溃不可复现（旧 trainer），当前 PPO 仍会静默劣化防守（放铳 8.3%→14.7%）。
- 固定参赛 Elo 方差 60–70 分，单次排名不可当结论（`CEILING_SUMMARY_2026-11-27.md`）。