# 能力盘点（2026-11-05 复核）：riichi-rl 现在到底会什么

> 背景：一份外部审阅基于「主会话上下文」给出 6 条判断，多处标注「需你验证」。
> 本文用文件阅读 + 实际跑命令逐条复核（不引用上下文猜测），并修掉复核过程中
> 暴露的 1 个真 bug（`eval_hanchan.py` 开箱即崩）。
> 结论先看：**引擎能完整对战；规则基线稳定；模型能加载能打但不会自己立直/和牌。**

## 0. 复核方法（可复现）

```bash
cd /mnt/e/riichi-rl
PY=/mnt/d/WorkPlace/my_rl_project/venv/bin/python
$PY -m tests.run_all                      # 66/66
$PY play_model.py --games 1 --quiet       # 模型能加载并打一局
$PY eval_hanchan.py --hanchans 3          # 模型 vs 随机 vs 向听贪心（本文修复后）
$PY engine_stats.py --games 200           # 规则级 200 局读数
$PY arena_elo.py --games 600 --seed 0     # 11 个策略/模型的 Elo（约 5 min）
# arena.py 是库不是 CLI：用 python -c 调 arena.evaluate(...).report()
```

## 1. 逐条核对

### ① 牌河顺序 — 记录有顺序，但数值特征里没用上（审阅说「未实现」只对一半）

- **数据层有顺序**：`Player.river` 是 list，弃牌时 `p.river.append(tile)`
  （`mahjong/env.py:822`）；被吃碰时 `_consume_from_river` 只 `pop()` 最后一张
  （`mahjong/env.py:901-906`），顺序不会被破坏。
- **人看的界面保留了顺序**：`llm_player.py:51`、`server.py:51` 把牌河按顺序拼成
  字符串喂给 LLM；观战页同样按顺序渲染。
- **神经观测没用顺序**：`mahjong/obs2.py` 的 5–8 平面、18 平面是「每种牌出现
  几次」的计数（`obs2.py:48/68/90`），只有 `last_discard` 一张 one-hot
  （`obs2.py:86`）；`opponent_model.py` 的特征同样是 34 维计数。
- **结论**：审阅说「若只记集合而无顺序，则未实现」→ 记录层是**有序的**；
  真正的缺口是「牌河顺序没有进策略特征」。要做需新增 obs 特征维度，
  会改变 774 维观测 → 旧 checkpoint 不兼容，需新开 obs 版本并重训。
  **这是设计项，不是补个字段。**

### ② 危险张判断 — 手写危险张被项目铁律**禁止**；学出来的也还没有

- **明确禁止**：`rules/RULES.md §15`「引擎里禁止出现……任何形如『安全牌』
  『危险牌』的标记」；`mahjong/policies.py:3-11` 同铁律；
  `mahjong/situation.py:14` 自述「不含进张数、危险牌、期望值」。
- **已有的替代物**（都是「局势/统计」而非单张危险度）：
  - `mahjong/situation.py`：`SIT_DEFENSE`（有人立直 → 必须防）、弃听判定；
  - `mahjong/scenarios.py`：放铳惩罚 `W_DEAL_DEFENSE=2.5`、弃听安全奖励；
  - `arena.py` 统计**放铳率**（本次 80 局表：greedy 0%、call_first/random 2.5%）；
  - `opponent_model.py`（从牌河/副露猜对手手牌）与 `hand_predictor.py`
    （预测和牌概率）——它们是「学危险度」的原料，但**尚未被组装成危险张打分**。
- **结论**：审阅「未见危险张逻辑」属实，但定性要改：这不是漏写，是项目
  主动禁止手写牌效先验；缺口是**学出来的危险度评估**（可用 opponent_model
  的输出 + 放铳率标签训练），而不是加一张 if 表。

### ③ 和牌率 — 有实现、有实测读数（审阅只看到设计意图）

- 三套实现：`engine_stats.py`（逐局聚合）、`arena.py` 的 `report()`
  （和了率/放铳率/立直率/流局率/平均和了点 + 批次 CI）、
  `arena_elo.py`（每参赛者 win_rate）、`eval_hanchan.py`（模型 vs 基线）。
- **实测**（2026-11-04/05）：
  - 200 局 4×greedy 自对弈：自摸 10 / 荣和 54 / 流局 136 → **和牌率 32.0%**
    （`rules/ENGINE_STATS_2026-11-04.json`）。
  - `arena.evaluate` 80 局 4 策略混战：greedy 和了率 15.0% / 放铳率 0%
    / 立直率 35% / 流局率 85%；call_first、random、yaochu 和了率 0%。
  - `eval_hanchan --hanchans 3`（24 局/项）：random 0%、shanten-greedy 37.5%、
    **模型（a2c_r11_it5）8.33%**。

### ④ 切什么牌 — 规则贪心已修，模型能跑但弱

- 规则：`mahjong/policies.py` `GreedyTenpai`（向听最小化，平手比待牌；
  不宣言流局、不随机吃碰、能立直就立直），回归 `tests/test_policies.py`。
- 模型：`play_model.py` 可跑（实测加载 `riichi_net.pt` 打 1 局 → 流局）；
  `eval_hanchan.py` 可跑（8.33%）；`arena_elo.py` 同时加载 11 个参赛者
  （含 54 槽、47 槽、243 槽三条模型线 + 4 个规则基线），600 局稳定出表。
- LLM 路线：`llm_player.py` + `serve_game.py`（观战页可切每席策略）。

### ⑤ 多轮迭代 — 是，且比审阅看到的更晚

- git：`git log --oneline | wc -l` = **15** 个提交，最新 `887ef65`（11-04 引擎
  大会战：19 处修复）；
- 测试：`tests/run_all` **66/66**（审阅写的 55/55 是 11-04 早些时候的读数；
  本次复核修 `eval_hanchan.py` 后新增 1 个 ckpt 加载回归用例），
  另有 `selftest.py` 15/15（T1–T12，含 200 局守恒）；
- 脚本：train_{reinforce,ppo,bc,dagger,hanchan,from_games}、eval_{model,
  hanchan,student}、arena{,_elo,_model}、serve_game/server、bench/ 等。

### ⑥ 能不能对战 — 能，三条路径都跑通了

| 路径 | 命令 | 实测 |
|---|---|---|
| 引擎规则对战（多策略+CI） | `python -c "import arena; print(arena.evaluate([...]).report())"` | 80 局表 + 两两 95% CI（见 §1③） |
| 模型/策略 Elo 对战 | `arena_elo.py --games 600` | 11 参赛者、310 s；bc_v1 1667 / greedy 1625 / from_games_v2 1589（总和 16500 零和） |
| 模型 vs 基线半庄评估 | `eval_hanchan.py --hanchans 3` | random 0% / greedy 37.5% / 模型 8.33% |
| 实时观战（浏览器） | `serve_game.py --seed 1000 --opp greedy_tenpai` → `http://localhost:8776/mahjong_view.html` | WS:8775 + HTTP:8776 启动正常；可中途切换每席策略 |

**限制**：模型「能对战」靠的是 `arena_elo.py` 里的**规则接管**（能荣和/自摸/
立直就由规则选，否则模型 argmax）。纯模型自身和牌率 8.33%（vs 随机 0%、
向听贪心 37.5%），仍不会主动立直/和牌。

## 2. 复核过程中修掉的真 bug

`eval_hanchan.py` 开箱即崩（审阅建议「先跑 eval_hanchan」正好踩中）：

1. **checkpoint 尺寸不匹配**：`load_model` 固定按 54 槽 `MahjongNet` 建网，
   但默认 glob 选中的 `a2c_r*_it*.pt` 是 **47 槽半庄模型** →
   `RuntimeError: size mismatch ... [47,256] vs [54,256]`。
   修复：按 `train_hanchan.N_ACTIONS(47)` + `touch_dim=TOUCH.TOUCH_DIM` +
   `extra_scalar=SIT_DIM(12)` 建网（与 `arena_elo.HanchanAgent` 一致）。
2. **默认权重取旧**：`sorted(glob)[-1]` 字典序把 `a2c_r9_*` 排在 `a2c_r10_*`
   之后；改为按 mtime 取最新。
3. **基线是旧的坏逻辑**：`play_shanten_greedy` 自己手写「见 RYUUKYOKU 就选」，
   一进听就宣言流局 → 和牌率恒 0（与已修 `GreedyTenpai` 不一致）。
   改为直接复用 `mahjong.policies.GreedyTenpai`。
   修复前读数：greedy 0/16；修复后：9/24=37.5%。

## 3. 结论（一句话版）

**能对战、有排名、有和牌率：`arena_elo.py` 600 局 + `eval_hanchan.py` + `serve_game.py`
三条路径都实测跑通；规则基线（向听贪心）和牌率 37.5%，纯模型 8.33%，
说明瓶颈在「网络不会主动立直/和牌」，不在引擎。**

## 4. 下一步（按投入产出）

1. **模型立直/和牌**：现在靠规则接管；要么训练时给立直/和牌动作加权
   （或 DAgger 引入规则教师），要么接受接管但把「接管」显式写进 Elo/评估口径。
2. **牌河顺序特征**：新增 obs 版本（序列/位置平面），旧 ckpt 不兼容，需重训。
3. **学出来的危险张**：用 `opponent_model` 输出 + 放铳标签做监督/奖励整形；
   注意不违反 §15（手写牌效禁令）。
4. 抢杠接响应阶段（`RULES.md §17 A17`）。