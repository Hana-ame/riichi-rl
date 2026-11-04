# 引擎修复记录：和牌率 2% → 24%（2026-11-04）

> 目标：找出"模型/规则 AI 能听牌却几乎不和牌"的根因。
> 结论：**不是模型问题，是引擎和策略共 5 处真 bug**。修复后 `greedy_tenpai`
> 实测和牌率 **2% → 24%**（50 局 12 和，全部自摸；立直 83 次）。
> 回归测试见 `tests/test_env_core.py`、`tests/test_policies.py`。

## 0. 排查顺序与读数

| 步骤 | 读数 | 结论 |
|---|---|---|
| 基线（修前） | 200 局 3 和 = **1.5%** | 真实麻将 40–50%，差 30 倍 |
| 玩家能否到听牌 | 30 局 17 局有人到向听 0 | 能听牌 |
| 14 张和牌型出现次数 | 40 局 1076 个决策点 **0 次** | 从没摸到和牌张 |
| 13 张真听牌次数 | 40 局仅 **5 次**（0.12/局） | 极低 |
| legal 含 RON / 实际选了 / 实际结算 | 21 / 21 / **1** | 选了却不结算 |
| 每局摸牌次数 | 仅 **11 张**（正常 ~70） | 对局被过早终止 |

## 1. 五处缺陷

### Bug 1：摸牌不被记录，自摸判定靠「最大牌号」猜
- 位置：`mahjong/env.py` `_draw_tile` / `_can_tsumo`
- 现象：注释自认"引擎没记录哪张是刚摸的，用最大 id 近似"；剔除错牌后
  牌型/役判定失败。
- 修复：`Player.last_drawn` 记录真实摸牌，`_can_tsumo` 使用它。
- 回归：`test_last_drawn_recorded`、`test_can_tsumo_uses_last_drawn`

### Bug 2：合法动作层不查「有役」，生成假和牌动作
- 位置：`mahjong/env.py` `_can_ron` / `_can_tsumo`
- 现象：`_can_ron` 只调 `REF.can_win_with`（牌型），而 `_finish_ron`
  要求 `_best_score` 有役；无役 RON 生成 → 策略选中 → 执行时
  `return self._advance()` 静默丢弃。实测 21 次选 RON 只结算 1 次。
- 修复：`_can_ron/_can_tsumo` 在牌型成立后追加有役检查（与执行层一致）。
- 回归：`test_ron_can_win_and_settles`、`test_no_yaku_ron_not_generated`

### Bug 3：未听牌时 GreedyTenpai 等于随机打牌
- 位置：`mahjong/policies.py` `GreedyTenpai.choose`
- 现象：只比较"打完后待牌种数"，未听牌时该值恒 0（原始 `best_score=-1`，
  所有候选并列），等于随机。
- 修复：按**向听数最小化**选牌（平手再比待牌数）。
- 回归：`test_picks_min_shanten_discard`

### Bug 4：听牌后立刻主动宣言流局
- 位置：`mahjong/policies.py` `GreedyTenpai.choose`
- 现象：`RYUUKYOKU` 在听牌时合法且策略优先级最高，一进听就结束本局，
  摸牌次数掉到 11 张，和牌机会归零。
- 修复：删除该分支（继续打牌等自摸/放铳）。
- 回归：`test_never_declares_ryuukyoku`

### Bug 5：响应阶段随机吃碰
- 位置：`mahjong/policies.py` `GreedyTenpai.choose`
- 现象：`return self.rng.choice(legal)` 会随机吃/碰，毁掉门前清
  （不能立直），立直率为 0。
- 修复：响应阶段只和牌（RON），其余 PASS。
- 回归：`test_response_always_pass_unless_ron`

## 2. 另外补上的策略改进

- **立直优先**：能立直就立直（听牌锁定 + 必然有役），
  回归 `test_riichi_preferred`。

## 3. 修复后读数

| 指标 | 修前 | 修后 |
|---|---|---|
| 和牌率（greedy 自对弈 50 局） | 2.0% | **24.0%** |
| 13 张真听牌（100 局） | 5 次 | **133 次** |
| 每局摸牌次数 | ~11 | 15.2–15.8/家 |
| 立直 | 0 | 1.66 次/局 |

## 4. 副产品发现（已顺手修）

- `train_reinforce.slot_of` 对 `tile=None` 抛 `TypeError`（PASS 类动作），
  已加防御并写测试 `test_all_slots_in_range`。

## 5. 未解决 / 后续

- 纯神经网络蒸馏模型仍不会主动立直/和牌（立直样本占比 2.4%，argmax 不选；
  8241 样本 vs 1.65M 参数过拟合）。当前观战用「规则接管立直/和牌 + 模型打牌」。
- 荣和（ron）在修复后样本中偏少，`_can_ron` 的有役检查是否过度严格
  待更大样本验证（下一步：200 局专项统计）。
---

# 第二轮审查：牌桌流程（同日续，2026-11-04 深夜）

> 触发：用户指令「你对牌桌逻辑代码也仔细看一眼」。第一轮只修到「能和牌」，
> 第二轮逐函数审查 `env.py` 的摸打/响应/结算/场次流程，又抓到 9 处真 bug。
> 关键读数：**和牌率 24% → 32%，荣和（ron）从 0 → 0.27 次/局**；
> 200 局点数守恒违规 **0**；`tests/run_all` **60/60**、`selftest.py` **15/15**。

## 6. 第二轮修复清单

### Bug 6：立直玩家被排除在响应阶段 → 立直后永远不能荣和
- 位置：`mahjong/env.py` `_step_discard` 的 `self.pending` 构造。
- 现象：`pending = [q for q in players if q.idx != p_idx and not q.riichi]`，
  立直者不进响应阶段；而 `_legal_response_phase` 里立直分支明确有 RON。
  结果：一手立直后只能自摸，放铳和恒为 0（修前 50 局样本全自摸）。
- 修复：立直者进响应阶段，但 `_legal_response_phase` 对立直者只给 RON/PASS
  （不能吃碰杠）。
- 回归：`tests/test_env_core.py::test_ron_can_win_and_settles` 等荣和用例；
  实跑读数见 §7。

### Bug 7：立直后仍能换牌（摸切锁手牌失效）
- 位置：`mahjong/env.py` `_legal_discard_phase`。
- 现象：立直后合法动作仍是「打任意一张」，等于可以换听/换牌，破坏立直规则。
- 修复：立直者只生成「打刚摸的那张（摸切）」+ 自摸，非法时兜底打最后一张。

### Bug 8：剩 4 张牌就提前流局（对局被砍到 ~15 张摸牌）
- 位置：`mahjong/env.py` `_advance`。
- 现象：`if len(wall) - wall_idx <= 4: _finish_ryuukyoku(reason="kyuushuu")`
  ——把「牌山剩 4 张」当成九种九牌/荒牌，实际每局只摸到 ~11–15 张。
- 修复：改为牌山耗尽（`<= 0`）才荒牌；局数上限仍由 `MAX_TURNS * 4` 兜底。

### Bug 9：振听判定拿 14 张去判（多一张，误判/漏判）
- 位置：`mahjong/env.py` `_update_furiten`。
- 现象：摸牌后 `full_counts()` 是 14 张，直接拿去调
  `can_win_with`（约定 13 张），判定不可信 → 振听状态随机。
- 修复：14 张时先去掉 `p.last_drawn` 回到 13 张再判。

### Bug 10：立直供托（kyokuten）消失 → 点数不守恒
- 位置：`mahjong/env.py` `_finish_tsumo` / `_finish_ron` / `GameResult`。
- 现象：立直扣的 1000 点进了 `self.kyokuten`，但和牌者的 `points_delta`
  不含供托，也没有字段把供托带出本局。实测 20 局里 19 局四家点数总和
  少 1000–2000（=桌上立直棒凭空消失）。
- 修复：`GameResult` 增加 `kyokuten` 字段；和牌时 `delta[winner] += kyokuten`
  并清零；`HanchanState` 增加 `kyokuten` 接力（和牌归零、流局保留）。
- 回归：`selftest.py::test_score_conservation` 判据改为
  `sum(scores) + kyokuten == 起手总点数 + 起手供托`。

### Bug 11：流局罚符分配错误
- 位置：`mahjong/env.py` `_finish_ryuukyoku`。
- 现象：旧实现按 300 点、且用 `seat_wind` 大小比较决定谁付谁，
  分支逻辑漏付/方向错误（实测 seed0 流局 delta 出现 `[-300,-300,+600,0]`）。
- 修复：按天凤规则「每个未听者向每个听者付 1000」：
  3 听 1 不听 → 不听者 −3000、3 家听者各 +1000；1 听 3 不听反之。

### Bug 12：杠后不翻宝牌指示牌
- 位置：`mahjong/env.py` `_do_kan` / `_do_minkan_from_discard` / `_flip_dora`。
- 现象：`dora_indicator` 恒为初始 1 张，杠了没有新宝牌，杠收益模型缺失。
- 修复：每次杠追加 1 张指示牌（上限 5 张，规则上限），
  `_dora_set()` 遍历全部指示牌；`check_tile_conservation` 同步计入。

### Bug 13：14 张判听牌默认去掉「最小牌号」的牌
- 位置：`mahjong/env.py` `_is_tenpai`。
- 现象：`total == 14` 且未指定 `extra` 时，旧实现丢 `min(tile)`，
  等于用错误的 13 张判听牌 → **弃牌阶段永远判不出听牌**，
  `RYUUKYOKU/NAGASHI` 动作从不生成（回归用例
  `test_never_declares_ryuukyoku` 抓到的真实缺陷）。
- 修复：默认去掉 `p.last_drawn`（摸切）；没有摸牌记录才退回最小牌号。
- 回归：`tests/test_env_core.py::test_tenpai_check_uses_last_drawn`。

### Bug 14：半庄点数不接力（每局从 25000 重开）
- 位置：`mahjong/env.py` `__init__` / `reset` / `_settle`，`mahjong/hanchan.py`。
- 现象：`HanchanState.sticks` 只在半庄层记流水；`MahjongEnv` 每局固定
  `RS.STARTING_POINTS`，且 `_settle` 只算 `scores` 不回写 `players[i].points`。
  → 立直资格（<1000 不能立直）、观测里的点数特征、观战界面的持越点全部失真。
- 修复：`MahjongEnv(start_points=...)`；`reset` 按起手点数建玩家；
  `_settle` 回写 `players[i].points = scores[i]`；
  `HanchanState.new_env` 传 `sticks`，`advance` 用 `result.scores` 更新
  `sticks`（`points_delta` 不含立直扣款，只加 delta 会让「sticks+供托」
  不再等于 100,000；旧式无 `scores` 的 result 保留 delta 回退路径）。
- 回归：`tests/test_hanchan.py`（5 例，含点数接力/供托/旧式 result 兼容）。

### Bug 15（口径）：统计/排名用 delta 会漏立直成本
- 位置：`mahjong/env.py` 新增 `net_points(result, start_points)`；
  `arena_elo.py` / `arena_model.py` / `eval_model.py` / `es_train.py` 改用。
- 说明：`points_delta` 是转账流水，立直时直接扣的 1000 点不在里面；
  「本局净变化」= `scores - 起手点数`。排名/平均点必须用后者。

## 7. 200 局专项统计（`greedy_tenpai` × 4 自对弈）

脚本：`engine_stats.py --games 200`；原始读数：
`rules/ENGINE_STATS_2026-11-04.json`（含逐局记录）。

| 指标 | 读数 |
|---|---|
| 局数 / 完成 | 200 / 200（39.6 s） |
| 自摸（tsumo） | 10（5.0%） |
| 荣和（ron） | **54（27.0%）** |
| 荒牌流局 | 136（68.0%） |
| 和牌合计 | 64 / 200 = **32.0%** |
| 平均摸牌次数 | 64.59（上限 72） |
| 立直 | 279 次 = 1.40/局 |
| 摸牌后听牌 | 1292 次 = 6.46/局 |
| ron 合法 → 结算 | 55 → 54（1 例被「一炮多响只取最近一家」吞掉，规则如此） |
| 牌型可和但 `_can_ron=False` | **0**（无役 0、振听 0） |
| 点数守恒违规 | **0** |
| delta 守恒违规 | **0** |

**对 §5 遗留问题的回答**：`_can_ron` 的有役检查**没有过度严格**——200 局里
「牌型成立但判无役/振听」为 0 次；ron 偏少的真正原因是旧引擎把立直者排除在
响应阶段（Bug 6），修复后 ron 占和牌的 84%（54/64）。

## 8. 仍未解决（明确写清，不假装完成）

- **纯神经网络蒸馏模型仍不会主动立直/和牌**（立直样本占比低，argmax 不选），
  观战/竞技仍用「和牌/立直规则接管 + 模型打牌」。
- **一炮多响**按 `RULES.md §9.2` 自决项实现为「逆时针最近一家」，不是多家和了。
- **王牌/岭上/四风连打/九种九牌** 按 `RULES.md §17` 自决项简化（未实现）。
- **半庄连庄**按本项目 `RULES.md §9.4`（和了者成庄、每局推进）实现，
  与标准立直麻将的「亲和了/亲听牌重复」不同；由 `selftest T12` 锁定，
  若要改成标准规则需同时改 `hanchan.py` 与 T12。
- 新的 200 局基线里荒牌率 68%，高于真实麻将（~50–60%），主因是
  `greedy_tenpai` 只做向听最小化、不做押引与防守；这不是引擎 bug。

## 9. 观战/竞技场侧另外三处（同日收尾）

### Bug 16：观战服务只打一局就退出
- 位置：`serve_game.py` `run_game`。
- 现象：外层循环 `while result is None and turn < 8`，而 `turn` 是**决策计数**
  （一局结束已 ~70），不是局数。第一局打完外层条件立刻为假 → 打印
  `[game over] 未结束`，半庄从不超过 1 局；`main()` 的 `while True` 又开
  新半庄，观战看不到东 1→东 2 的持越点/供托接力。
- 修复：改用 `deals` 计已完成局数（上限 8），每局结算后 `+1`。
- 验证：headless 跑 `run_game(seed=1000)`，连续 8 局正常推进，
  结束打印 `[game over] ryuukyoku`（修复前是 `[game over] 未结束`）。

### Bug 17：Elo 四人更新不是零和 → 全体评分漂移
- 位置：`arena_elo.py` `elo_update`。
- 现象：把名次分 `[0.75, 0.5, 0.25, 0]` 直接当**逐对**得分：一对
  (第 1, 第 4) 的 `s_a + s_b = 0.75 ≠ 1`，每局总分不守恒。实测 200 局后
  全场均值从 1500 掉到 ~870；名次排序看似正常，但 Elo 刻度整体偏移，
  不能跨批次比较。
- 修复：逐对用「名次比较」的 1/0.5/0 计分（零和）。
- 回归：`tests/test_arena.py`（2 例：单局零和、重复对局收敛且总和不变）。

### 口径：在线推理线程数按 `MODEL_BENCH.md` §4 落地
- `cpu_only.py` 新增 `set_inference_threads(n=2)` / `set_batch_threads(n=6)`；
  `serve_game.py`、`arena_elo.py` 启动时设为 2 线程，并改成「先 import
  `cpu_only` 再 import torch」，确保 CPU-only 约束真的先于 torch 初始化。
- 实测：Elo 竞技场 200 局，12 线程跑 >11 分钟仍未完成（单样本 10–30 ms
  且有 0.5 s 长尾），**2 线程 93 秒跑完**；趋势与 MODEL_BENCH §4 一致。

## 10. 规则文档层面的两处错误（同日收尾，影响玩法本身）

审查「牌桌逻辑」不能只对代码，还要对 `rules/RULES.md` 的规则本身。
发现规范文档有两处把标准立直麻将写反/写错，代码是照错的规范实现的：

### Bug 18：振听方向写反（振听禁的是荣和，不是自摸）
- 位置：`rules/RULES.md` §11 原文、`mahjong/env.py` `_can_tsumo` / `_can_ron`。
- 现象：旧规范写「振听后**不得自摸**（放铳和了仍可成立）」，代码照做：
  `_can_tsumo` 里 `if p.furiten: return False`，`_can_ron` 只对抢杠查振听。
  与标准立直麻将正好相反——标准是**振听不能荣和（含抢杠），自摸照常**。
- 修复：`_can_tsumo` 移除振听拦截；`_can_ron` 对普通放铳也拦截振听；
  `rules/RULES.md` §11 改写并标注修正记录。
- 回归：`tests/test_env_core.py::TestFuritenRules`（2 例）、
  `selftest.py` T6/T9 改判据、T2 非法和了校验改为「振听荣和=非法」。
- 影响：荣和率会下降（振听局面不再能白捡放铳），这是规则正确性优先。

### Bug 19：`RULES.md` §12.7 把「牌山剩 ≤4 张」当成九种九牌/荒牌条件
- 位置：`rules/RULES.md` §12.7（原本就是旧 `_advance` 提前流局的依据）。
- 现象：规范写「牌山剩 <= 4 张时进入流局检查」，被代码直译成每局只摸
  11–15 张就结束（见 §6 Bug 8）。
- 修复：§12.7 改为「荒牌 = 牌山摸尽」；九种九牌单列为**不实现**
  （§17 新增 A16）。代码已是「牌山耗尽才荒牌」。

## 11. 补记

- **振听方向修正后重跑 200 局读数不变**（tsumo 10 / ron 54 / 流局 136）：
  这批 `greedy_tenpai` 自对弈里没有出现「振听却可荣和」的局面，
  所以该修正不改变基线，但会改变更复杂策略（尤其人类/BC）的荣和率，
  属于规则正确性修正而非指标优化。
- **抢杠（chankan）仍未实现**（新发现，写入 `RULES.md §17 A17`）：
  `_step_discard` 的加杠路径直接 `_do_kan → _after_claim` 成立，
  不进入响应阶段；`_can_ron(from_chankan=True)` 只有单测在调用。
  要补齐需要：加杠后开响应阶段（last_discard=加杠牌）+ `scoring.evaluate`
  增加 `is_chankan` 并计 1 番 + 加杠失败回滚。
