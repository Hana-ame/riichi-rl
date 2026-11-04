# 显式预测模块与对战采样规则——证据与口径（2026-11-05）

> 审阅说「危险张/和牌率无独立预测模块、只有事后统计」「较好一半+随机替换未见
> 文档化」。前者不准确（有模型，但未作为策略主头），后者已实现在 arena_elo；
> 本文给文件/行号/用法，避免再被当成“没做”。

## 1. 和牌率：有显式预测模型（不是只有事后统计）

- `hand_predictor.py`：三输出头
  - `win`：本局最终和牌概率（sigmoid）
  - `pts`：若和了的预期点数（回归）
  - `turns`：还几巡能听牌（softplus）
- 训练数据：`data/selfplay` 703 局 → 51,666 样本；`checkpoints/hand_predictor.pt`
  （val_win≈0.052、turns MAE 14.3 巡——质量弱但模型存在）。
- 已接入观战：`serve_game.py:329 hand_predictions()` 每帧输出四家
  `{win, pts, turns}`，在 `state_json` 的 `hpred` 字段。
- 用法：
  ```python
  import serve_game as SG
  SG.hand_predictions(env)   # {0..3: {win, pts, turns}}
  ```
- 局限：模型只在**观测服务**里跑，没有把 win_prob 作为策略决策的额外输入
  （当前策略是端到端 argmax）。要做「按和牌率决策」需把它接进 policy 或做
  辅助头联合训练——这是 P3 之后的事。

## 2. 危险张：有两级显式预测

- **对手持牌概率**：`opponent_model.py::OpponentNet`（108→34 sigmoid），
  输入含目标家牌河 counts/副露/立直/巡目等；重训后听牌样本 **top5 0.368**
  （随机 0.147）；`serve_game.py:251 get_opp_model / :271 opp_predictions` 集成。
- **对手和牌张概率**：`danger_model.py::DangerNet`（108→34 sigmoid），
  标签=听牌时 `can_win_with` 枚举的和牌张；65,696 样本、听牌样本
  **top5 0.882**；用于 `hybrid_agents.py:248 HybridShantenWinDangerAgent`
  （`hybridw:` 前缀）的弃牌 tie-break。
- 决策接入：`hybrid_agents.py:147 HybridShantenDangerAgent` 用持有概率×威胁
  权重做危险分，`hybridd:` 已在版本基准与 500 局验证里使用。
- 局限：危险张是**tie-break/辅助信号**，没有作为主策略的独立预测头
  （policy 的输入里没有 danger 通道）。这是「有预测、未主头」。

## 3. 对战采样规则「表现较好一半 + 随机替换」

- **已在 `arena_elo.py` 实现并生效**：
  - `--warmup-games 100`：全体热身；
  - 之后每 `--reselect-every 100` 局：保留 Elo 前一半（`len//2` 名）+
    从落选者随机 `--random-fill 3` 个；
  - `rules/ELO.md` 头部保存最近一次选员名单；
  - 生效痕迹：被换下者局数明显少（dagger 29、from_games_v1 67 vs 常驻 219–230）。
- **`version_ladder.py` 不是选员器**：它评估的是「固定版本池的晋级闸门」
  （同池随机 4 人、相邻版本净点差 CI）。若把「前一半+随机替换」塞进天梯，
  PASS/FAIL 的对比对象会变、闸门失去意义。两者职责分开是对的。
- 若要在 `VERSION_REGISTRY` 展示：可在 registry 增加「arena 参赛局数/
  arena Elo」列（数据已在 ELO.json），属于展示项，下一步可加。

## 4. 本轮结论

| 审阅说法 | 实际 |
|---|---|
| 和牌率无显式预测模块 | ❌ 有 `hand_predictor.py`（win prob 三头 + 已接观战）；只是未作为策略主头 |
| 危险张无独立模块 | ❌ 有 opponent_model（持有概率）与 danger_model（和牌张概率），hybridd/hybridw 已接入决策；非主头 |
| 「较好一半+随机替换」未文档化 | ❌ 实现在 `arena_elo.py`（参数+ELO.md 记录+局数痕迹）；天梯按设计不做选员 |

## 5. 待做（与审阅优先级一致）

1. 和牌率：DAgger 聚合数据 ≥3 epochs + 采集扩到 500–1000 局（进行中）；
2. 副露/杠独立策略（P3）：call_first 教师标签 + 特殊槽加权；
3. dora 独立策略（P4）：aux_tasks 加 dora 任务 + `--aux-weights`；
4. 把 win_prob/danger 作为**策略输入通道**（辅助头/特征）——本轮未做。
