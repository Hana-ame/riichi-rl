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
---

# 附：第二轮审阅复核（2026-11-05 00:36）

外部第二轮审阅仍基于上下文推断，三条需修正/补证，另发现一个新缺陷。

## 1. 牌河顺序：不是「集合」，是有序 list（现场演示）

```
$ grep -n "river" mahjong/env.py
94:        self.river = []          # Player 初始化
822:        p.river.append(tile)     # 弃牌入河（append 保序）
902:            src.river.pop()      # 被吃碰时只移除尾张

# 实跑：seed=7，14 步后
river[0] = [0, 6]        # 1m 后 7m
river[1] = [0, 6]
log 前 8 条 = [(0,0,'discard','1m'), (1,1,'discard','1m'), (2,2,'discard','1m'),
             (3,3,'discard','5p'), (4,0,'discard','7m'), ...]
```

结论同第一轮：**记录层有序；数值特征（obs2 的 5–8/18 平面）只做每牌计数，
只有 `last_discard` 一张 one-hot**。要「按牌河顺序推断手牌形状」必须新增
位置/序列特征（会改 774 维观测，旧 ckpt 不兼容）。

## 2. 「无任何放铳率/攻防判定」不成立

- `arena.py:65,71,178` 有 **放铳率**（`deal_in_rate`）报告；
- `mahjong/scenarios.py` 有放铳惩罚 `W_DEAL_BASE=1.0`、`W_DEAL_DEFENSE=2.5`、
  弃听安全奖励；
- `mahjong/situation.py` 有 `SIT_DEFENSE`（有人立直 → 全防）与弃听判定 `deserves_give_up`。

准确表述：**有「局面级攻防模式 + 放铳率统计」，没有「单张危险度打分」**；
后者按 `RULES.md §15` 不许手写，应由模型学。

## 3. 「逐手/逐局面和牌率预测未见证据」→ 现在有模型了（本轮修好并训练）

`hand_predictor.py` 三输出头（`win_prob` / 期望点数 / 还几巡听牌），
原先训练**必崩**（两个 bug，本轮修掉）：

1. `int(d["winner"])` 遇流局 None → `TypeError`（703 局里 97 局是流局）；
   已加 `winner_index()` 读 `None → -1`，回归 `tests/test_models.py`。
2. 设备判定 `cuda if available` 违反项目 CPU-only 约束，且验证 target
   没搬设备 → `cuda:0 vs cpu` 报错；改为 `dev="cpu"` 并搬 target。

训练结果（`--epochs 20`，CPU，51,666 样本 / 703 文件 / 跳旧格式 97）：

```
baseline_win=0.009  val_win=0.0516  val_pts_mae≈500点  val_turns_mae=14.32巡
saved checkpoints/hand_predictor.pt
```

已接入观战：`serve_game.hand_predictions(env)` 实测返回四家
`{win≈0.01, pts=0, turns≈0.9}`。**质量仍弱**（win 头接近常数、巡数头塌到均值），
只能当特征/参考，不能当判据。

## 4. 新发现：54 槽线的动作槽映射不是单射（模型叫牌会被吞）

`train_reinforce.slot_of` 是哈希：`34 + ((type*7 + tile) % 20)`。
PASS 固定槽 50；而 CHI(起始牌) / PON(牌) 在特定 tile 上也算出 50。
实测 40 局 10,572 个决策点：

```
有槽冲突的决策点 66（0.62%），涉及 32 局
冲突组合：CHI+PASS 53 次、PON+PASS 13 次
```

`legal[slots.index(pick)]` 取「列表里第一个同槽动作」，而 `_legal_response_phase`
把 PASS 放在最前 → 模型即使想 CHI/PON，也会被执行成 PASS；训练数据里这些
CHI/PON 的标签同样被记成 50（PASS）→ 训练标签被污染。
（RON 撞 PASS 时因为 RON 被 `insert(0)` 放最前，反而保住；TSUMO 在 RIICHI 前。）

**这正是「54 槽线模型不自给自足」的结构性原因之一**。47 槽线（`train_hanchan`）
用的是**固定特殊槽**（TSUMO=34/RON=35/RIICHI=36/…），不存在该问题；实测
47 槽 `a2c_r11_it5` 在 `eval_hanchan` 里**不带规则接管**也能和 2/24（8.33%）。
→ 「让神经模型自己会和牌」的正确路线：**固定槽位 + 特殊动作加权/DAgger**，
不要继续修哈希槽线。

## 5. 第二轮其余判断的复核

- ⑤ 多轮迭代：`tests/run_all` 已 **67/67**（第一轮复核后新增 2 个回归）。
- ⑥ 「模型本尊尚弱」成立：`eval_hanchan` 模型 8.33% vs 向听贪心 37.5%；
  600 局 Elo 里非 bc 神经模型和牌率 0–3.4%（54 槽线 0%；47 槽线 1.9–3.4%）。
- 优先做「让模型自己会和牌」同意；具体抓手按上面 §4：先换固定槽位（新数据
  生成 + 重训），再做特殊动作加权/DAgger，评估用 `eval_hanchan`（不带接管）。

---

# 第三轮复核证据：副露/吃碰杠、dora、牌河→手牌形状（2026-11-05）

审阅要求对三项补「文件+行号+是否参赛」的直接证据。均为现场 grep/实测。

## 3.1 副露（吃/碰/杠）策略

**规则层（引擎，合法动作 + 执行）**
- `mahjong/env.py:679` 吃（仅上家、字牌不可）、`:685` 碰、`:689` 明杠、
  `:694` 加杠；`:414` 暗杠、`:420` 加杠（打牌阶段）；`:879` 响应裁决顺序
  （和了 > 吃 > 明杠/加杠 > 碰）。
- 杠后`_flip_dora()`（`:988`、`:1012`）追加宝牌指示牌。

**策略层（谁真的吃碰杠）**
- `mahjong/policies.py:274-281` `CallFirst._PRIORITY`：明杠 50 / 加杠 50 /
  碰 40 / 吃 30 / 暗杠 5（都会主动副露）。
- `GreedyTenpai`（policies.py:~86-98）响应阶段**只荣和、否则 PASS**；`damaten`/
  `riichi_*` 继承同一逻辑 → 不吃碰杠（保门前清）。
- 47 槽半庄线包含副露/杠固定槽：`train_hanchan.py:58-74`
  `SLOT_PON=39 / SLOT_MINKAN=40 / SLOT_ADD_KAN=41 / SLOT_ANKAN=42 / SLOT_CHI=44..46`，
  因此 47 槽模型**可以学副露**。

**是否参赛 + 实测动作数**（40 局混战，直接统计实际选出的动作）:
```
call_first    总动作 2565  副露/杠 {'CHI':61,'PON':37,'ADD_KAN':11}
yaochu_first  总动作 2555  副露/杠 {'CHI':49,'PON':41,'MINKAN':1,'ADD_KAN':12}
greedy_tenpai 总动作 2556  副露/杠 {}          立直 25 荣和 9
damaten       总动作 2556  副露/杠 {}          立直 0  荣和 3
```
参赛证据：`rules/ELO.md` — `call_first` 156 局（放铳率 10.9%、平均点 −1111）、
`yaochu_first` 156 局；两者都在 600 局 Elo 表内。

**reward/场景**：副露没有独立奖励，收益通过点数（`scenarios.reward_for_result`
用 points_delta / 放铳惩罚）体现；没有「碰更好/吃更好」的权重（§15 禁）。
结论：吃碰杠**策略存在且已参赛**（call_first/yaochu_first），但神经网络模型
（除 teacher47 线）在 Elo 表里和牌率仍 0–3%，**副露学习成效未体现**。

## 3.2 dora 策略

- **输入特征**：`mahjong/obs2.py:78-103`——plan 15 宝牌指示牌 one-hot、plan 16
  里宝牌指示牌 one-hot、plan 20 宝牌集合 one-hot（所有 54/47 槽模型共享 obs2，
  即每个模型都“看得到” dora）。
- **规则/计分**：`env.py:286 _flip_dora`（初始指示牌 + 杠后追加）、
  `:599 _dora_set`、`:632 _dora_count`、`:638 _ura_dora_count`；
  `scoring.evaluate(..., dora_count, aka_count, ura_dora_count)` 计入番数；
  立直和了时 `_set_ura_dora()` 才开里宝牌。
- **hand_predictor**：`hand_predictor.py:51-57` `build_feature(..., dora)` 把
  宝牌 id 作为输入特征；`serve_game` 每帧传 `env.dora_indicator`。
- **策略层**：**没有独立的“dora 策略”**——规则策略不读 dora；神经网络通过
  obs 平面看到 dora，由训练自己决定怎么用；teacher47_v3 的「加权 BC」权重是
  **动作类型**（和牌/自摸/立直），不是 dora 权重。
- **是否参赛**：所有含 obs2 的模型（riichi_net/from_games/a2c/dagger/hc/
  teacher47_v3）都输入 dora 特征，均在 ELO.md 参赛；teacher47_v3 98 局。
  结论：dora **有特征+计分+杠翻宝牌**，但**没有显式 dora 策略**。

## 3.3 牌河 → 手牌形状/向听

- **记录层有序**：`env.py:829 p.river.append(tile)`（弃牌顺序），被吃碰时
  `_consume_from_river` 只移尾张。
- **观测层是计数不是顺序**：`obs2.py:49`（四家牌河计数平面 5–8）、`:91`
  （自家牌河平面 18）、`:87`（`last_discard` one-hot 平面 17）。→
  「**按牌河顺序**判断形状/向听」**没有实现**（顺序未进特征）。
- **对手手牌推断模型（成品，可运行）**：
  `opponent_model.py:36-46` `OpponentNet` 输出 **34 维每张牌持有概率**；
  `:49-66 build_features` 输入 = 观察者手牌 + **目标家牌河 counts** +
  其他两家牌河聚合 + 副露数/立直/已打张数 + 巡目/墙余/dora；
  服务集成 `serve_game.py:248 get_opp_model / :268 opp_predictions / :382`。
  **本轮重训**（CPU，703 局 → 139,761 训练 / 15,528 验证样本，10 epochs）：
  `val_loss 0.2431→0.2263`，**top5 命中 0.279→0.368**（随机基线 5/34=0.147），
  即从可见牌河推断对手持牌已显著优于随机。旧 ckpt（仅 786 样本）top5≈0.146
  即随机水平——现已不可用的问题解决。
- 结论：**牌河顺序→形状**未实现；**牌河计数→对手持牌概率**已有成品模型且
  重训后 top5=36.8%（可再从持牌概率读出形/向听，但不是“按顺序”）。

## 3.4 顺带修正

- 审阅「aux_tasks.py 已有 defense/give_up/last_chance」不成立：这三项在
  `scenarios.py`/`situation.py`；`aux_tasks.py` 是 7 项成就任务。
- 「模型预测和牌率未见」不成立：`hand_predictor.py` + 已训 ckpt + 服务集成。
- `opponent_model.py` 设备判断原为 `cuda if available`，违反项目 CPU-only
  硬约束，本轮改为 `dev="cpu"`（与 hand_predictor 同修）。
