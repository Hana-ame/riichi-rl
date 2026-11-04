# 功能状态证据核对（2026-11-05，回应审阅「条件不足」）

> 审阅说上下文未见：危险张、牌河顺序→形状、副露、dora、arena_elo、东风+半庄。
> 这些都是**早前已做、只是不在最近上下文**。以下给文件/行号/读数；并明确真正
> 未完成的部分与计划。

## 一、审阅列「未见」的 6 项：已有证据

| 项 | 状态 | 证据 |
|---|---|---|
| **危险张** | ✅ 已实现（2 版）+ 已做 500 局验证 | `hybrid_agents.py:147` `HybridShantenDangerAgent`（对手持牌概率×威胁权重）、`:248` `HybridShantenWinDangerAgent`（DangerNet 对手和牌张概率）；`danger_model.py::DangerNet`；`rules/DANGER_MODEL_2026-11-05.md`、`rules/DANGER_VALIDATION_2026-11-05.md`：500 局 hybridd 放铳 5.8% vs hybrid 7.6%（差 −1.80pt，CI 跨 0 → **INCONCLUSIVE**），场均 +774 vs −485 |
| **牌河顺序→形状** | ⚠️ 顺序未进特征（真）；但「牌河→对手持牌概率」有成品 | 顺序仍无（`obs2.py` 是计数+last_discard）；`opponent_model.py:36/49`（牌河 counts→34 维持牌概率），重训后 **top5 0.368**（随机 0.147），`serve_game.py:251/271` 集成；`rules/CAPABILITY_AUDIT_2026-11-05.md §3.3` |
| **副露策略** | ✅ 规则策略（有优先级、已参赛）；❌ 模型独立策略未做 | `mahjong/policies.py:274-281` CallFirst（明杠50/加杠50/碰40/吃30/暗杠5）；40 局实测 CHI61/PON37/ADD_KAN11（yaochu 另有 MINKAN1/ADD_KAN12）；ELO 600 局里 call_first 156 局；47 槽含 PON/CHI/KAN 槽（`train_hanchan.py:58-74`） |
| **dora 策略** | ✅ 特征/计分/杠翻；❌ 独立策略未做 | `obs2.py:79/83/100`（dora/里宝/宝牌集合三平面）；`env.py:286 _flip_dora`、`:599 _dora_set`、`:632 _dora_count`；scoring 计入番；`hand_predictor.py:51` 把 dora 当特征。无独立 dora 策略（网络自学） |
| **ELO 评估** | ✅ 已实现，600 局结果在库 | `arena_elo.py`（4 人随机桌、逐对零和 K=24、`--aux-weight`、放铳率列）；*选员* `:361-365 --warmup-games 100/--reselect-every 100/--random-fill 3`；结果 `rules/ELO.md`：from_games_v2 1692.8 / bc_v1 1666.3 / damaten 1651.3 / greedy 1597.2 … yaochu 1306.4（15 人总和 22500 零和） |
| **东风+半庄** | ✅ 已实现并分别评估 | `hanchan.py:31/118` `east_only`（东 4 即终局）、`eval_hanchan.py:120/125` `--mode {hanchan,east}`；实测 teacher47_v3 **半庄 12.5%(3/24) / 东风战 16.7%(2/12)**（`rules/GAME_MODES_2026-11-05.md`） |

## 二、审阅说对的：真正还缺的

1. **纯模型和牌率/听牌率不达标**：teacher47_v3 听牌率 20.0%（hybrid 38.8%）、
   和牌 3.8%；放铳率 3.8–6.2% 偏低是「没进听/进听难和」的结果
   （`rules/TENPAI_DEALIN_2026-11-05.md`）。
2. **牌河顺序序列特征**：未做，需要新 obs 版本（旧 ckpt 不兼容，要重训）。
3. **独立副露/杠、dora 学习策略**：未做（有规则/特征基础）。
4. **DAgger/数据扩充**：进行中——见第三节。

## 三、本轮正在做的：教师数据 30→100 半庄

- 已生成 `data/teacher47_100.npz`：**100 半庄 / 208,542 样本**
  （tsumo 70、ron 215、riichi 1173；生成 155s）。
- 正在后台训练 `checkpoints/teacher47_100h.pt`
  （`train_teacher_bc.py --data data/teacher47_100.npz --epochs 2
  --special-weight 20 --discard-weight 3 --pass-weight 0.1`，约 10 分钟）。
- 验收：`eval_hanchan --hanchans 10`（80 局）对比 teacher47_v3 同种子；
  再用 `version_ladder.py` 闸门（点差 CI）决定是否发布。

## 四、实现/验证计划（按优先级与 ETA）

| # | 任务 | 做法 | 验收 | ETA |
|---|---|---|---|---|
| 1 | 100 半庄 BC（进行中） | 208k 样本）+ 决策类型加权 | eval_hanchan 80 局：听牌率/和牌率 ≥ 基线；version_ladder PASS | ~15 min |
| 2 | DAgger 数据 | 模型自对弈 → GreedyTenpai 在模型状态打标 → 与 BC 数据混合重训 | 和牌率进 15–25%、听牌 ≥35% | 1–2 h |
| 3 | 独立副露/杠策略 | `call_first` 作教师生成带吃碰杠标签的数据 + 特殊槽加权 BC | 模型叫牌率接近规则、放铳率不恶化 | 1 h |
| 4 | dora 辅助任务 | `aux_tasks` 加 `dora`（和牌且 dora/赤/里宝≥1 番）+ `--aux-weights` 对照 | dora 和牌占比/场均提升 | 30 min |
| 5 | 牌河顺序 obs v2 | 序列/位置平面（新 obs 版本）→ 重训 | 对手形状推断指标（top5）提升 | 2–3 h |

## 五、复现命令

```bash
cd /mnt/e/riichi-rl
PY=/mnt/d/WorkPlace/my_rl_project/venv/bin/python
$PY tenpai_audit.py --deals 80                 # 听牌率/放铳率审计
$PY arena_elo.py --games 600 --seed 0          # ELO（含选员/放铳率/任务奖励）
$PY eval_hanchan.py --mode east --hanchans 3   # 东风战
$PY eval_hanchan.py --mode hanchan --hanchans 3
$PY gen_teacher_data.py --hanchans 100 --out data/teacher47_100.npz   # 208k 样本
```
