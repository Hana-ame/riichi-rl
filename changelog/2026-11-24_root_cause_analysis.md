# 立直麻将 AI（riichi-rl）四大核心问题根因分析报告

- **日期**：2026-11-24
- **范围**：针对「train 不出来、副露做不出、不懂弃胡、不懂流听」四大现象，对引擎规则层、特征编码层、动作空间、网络结构、奖励函数与训练管线进行全链路代码审计与实测验证。

---

## 0. 核心结论总览

| 现象 | 第一根因（决定性） | 次级根因（放大器） | 所在文件与行号 |
|---|---|---|---|
| **副露做不出** | 引擎 `enumerate_all_with_melds` 在任意副露下恒返回空列表 `[]`，**副露后数学上 100% 无法和牌**；且完全缺失「役牌」判定、副露 20 符被误杀 | 副露宝牌未计入、响应优先级 `CHI > PON` 写反、吃槽哈希 `tile % 3`、`obs2` 缺弃牌者相对座位与候选副露特征 | `mahjong/hand.py:149-179`<br>`mahjong/scoring.py:125-129, 226-275, 357-358`<br>`mahjong/env.py:566, 642-646, 890`<br>`train_hanchan.py:86-87` |
| **不懂弃胡** | `obs2.py` **完全未编码对手立直状态、立直宣言牌与现物（安全牌）平面**，网络对对手立直是「瞎的」 | `GreedyTenpai` 教师从不弃和、`shanten_weight * (sb - sa)` 惩罚拆牌防守、缺少半庄顺位分（Uma）目标、`danger_model` 跳过所有副露对手 | `mahjong/obs2.py:116-119, 144`<br>`mahjong/situation.py:95`<br>`mahjong/policies.py:61-160`<br>`train_ppo.py:199`<br>`mahjong/scenarios.py:64-65` |
| **不懂流听** | 引擎流局规则与余牌观测多处错位（2 听罚符 `±2000`、庄家流听不连庄、死牌山 14 张未扣除、杠后 `_is_tenpai` 异常） | 副露瘫痪导致无法终盘吃碰抢形式听牌、`deserves_give_up` 在后半局未听即标弃听、纯 MC 回报下方差淹没 `±0.1~0.3` 流听奖励 | `mahjong/env.py:305, 312, 483-506, 867, 1135-1138`<br>`mahjong/hanchan.py:96-107`<br>`mahjong/situation.py:134-158` |
| **train 不出来** | `env.py` 的 `_best_score` 未真正切换到开源算分器 `refengine`，仍在跑有严重 Bug 的自研算分；且 `refengine.to136` 映射有碰撞隐患 | `train_ppo.py` 实为无 Critic 的 REINFORCE；`train_hanchan.py` 缺 GAE($\lambda$) 与半庄顺位回传；纯 Python 吞吐仅 `0.4 g/s`；默认 `MahjongNet` 跨花色卷积 + `AdaptiveAvgPool1d(4)` 丢空间信息 | `mahjong/env.py:453, 523-558`<br>`mahjong/refengine.py:42-44, 64-130`<br>`train_ppo.py:190-235`<br>`train_hanchan.py:533-605`<br>`mahjong/net.py:54, 92` |

---

## 1. 副露做不出：引擎规则层直接锁死了副露和牌

### 1.1 实测复现（已验证）

直接在仓库根目录运行以下脚本测试 `MahjongEnv` 的和牌判定：

```python
from mahjong import env as E

env = E.MahjongEnv(seed=42)
p = env.players[0]

# 1. 副露断幺九（碰 2m + 345m 345p 345s 66s 自摸）
p.melds = [("pon", 1, 1)]
p.hand = [2, 3, 4, 11, 12, 13, 20, 21, 22, 23, 23]
p.last_drawn = 23
print("Open Tanyao _can_tsumo:", env._can_tsumo(0))          # 输出: False

# 2. 副露役牌（碰白板 + 234m 567p 234s 55s 自摸）
p.melds = [("pon", 31, 1)]
p.hand = [1, 2, 3, 13, 14, 15, 19, 20, 21, 22, 22]
p.last_drawn = 22
print("Open Yakuhai (Haku) _can_tsumo:", env._can_tsumo(0))  # 输出: False

# 3. 门清断幺九（对照）
p.melds = []
p.hand = [1, 1, 1, 2, 3, 4, 11, 12, 13, 20, 21, 22, 23, 23]
p.last_drawn = 23
print("Menzen Tanyao _can_tsumo:", env._can_tsumo(0))        # 输出: True

# 4. 门清役牌荣和（白板暗刻，不含门前清自摸）
p.melds = []
p.hand = [31, 31, 31, 1, 2, 3, 13, 14, 15, 19, 20, 21, 22]
env.last_discard = 22
env.last_discard_player = 1
print("Menzen Yakuhai (Haku) _can_ron:", env._can_ron(0))    # 输出: False
```

**实测结果**：只要手上有任何副露，标准型和牌判定恒为 `False`；无论门清还是副露，役牌（白/发/中/场风/自风）均无役。

### 1.2 逐项代码级根因

1. **`mahjong/hand.py:149-179` (`enumerate_all_with_melds`) 恒返回空列表**
   - 传入 `concealed_counts`（11 张暗牌，对应 3 组面子 + 1 雀头）与 `melds`（1 组副露）。
   - 代码执行 `need = 4 - len(melds) = 3`，但紧接着第 163–169 行把 `melds` 的牌**重新加进** `merged`：
     - 碰：`merged[t] += 3`，总牌数变成 `11 + 3 = 14`。
     - 吃：`size = 3; merged[t] += 3; merged[t+1] += 1; merged[t+2] += 1`，凭空多加了 2 张，总牌数变成 `11 + 5 = 16`。
   - 随后去掉 2 张雀头，拿 12 张（或 14 张）调用 `enumerate_standard(c, need=3)`。而 `_recurse` 终止条件是 `need == 0 and sum(c) == 0`。3 组面子最多消耗 9 张牌，`12 - 9 = 3 != 0`，**数学上永远无解**。
   - **直接后果**：`env._best_score()` 在有副露时永远返回 `None`，AI 一旦吃/碰/杠，就彻底丧失本局和牌资格。这也完全解释了 `rules/CALL_STRATEGY_2026-11-05.md` 中「无论叫牌阈值怎么调，call_agent 和牌率永远只有 no_call 的一半（5% vs 10%）」的异常现象。

2. **`mahjong/scoring.py:226-275` (`_detect_yakus`) 缺失役牌（白/发/中/自风/场风）**
   - `YAKU_HAN` 和 `_detect_yakus` 中只列了断幺九、对对和、平和、三色、混/清一色等，**完全没有三元牌与场风/自风刻子的役牌判定**。
   - 此外 `mahjong/yaku.py` 内部还有多处役种实现错误：如 `is_iippekou`（写成了 `len(set(seqs)) == 3`，实为一盃口）与 `is_ittou`（写成了 `max(c.values()) == 3`，实为三同顺）混淆；缺一气通贯、小三元、海底/河底、岭上、抢杠；`is_chunchan` 写成了要求雀头非幺九。

3. **`mahjong/scoring.py:125-129, 357-358` 误杀副露 20 符（喰いタン平和形）**
   - `compute_fu` 先执行 `fu += base`（副露 `base = 20`），后执行 `if is_open_hand and fu == 0: fu += 2`——此时 `fu` 已经是 `20` 而非 `0`，补 2 符永远不会触发。
   - 随后 `evaluate` 第 357 行：`if fu == 20 and han < 2 and not is_riichi: return ScoreResult([], 0, ...)` 直接判为无役。在日麻规则中，副露 0 符（全顺子 + 非役牌雀头 + 两面听）应补 2 符进位为 **30 符 1 番（1000 点）**。

4. **副露标记、宝牌与响应优先级错误**
   - `mahjong/env.py:566`：`is_melded = (kind, t, None) in [(m[0], m[1], m[2]) for m in p.melds]`。`m[2]` 是 `called_from`（0..3 的整数），永远不等于 `None`，导致所有副露在算符/三暗刻/四暗刻时全部被错标为「暗刻」。
   - `mahjong/env.py:642-646`：`_dora_count` 只统计 `p.hand`，**副露里的宝牌不算番**；`_aka_count` 对顺子副露只查 `m[1] == t`（起始牌），漏掉顺子中间/末尾的 5m/5p/5s。
   - `mahjong/env.py:890`：`_step_response` 按 `(ACT_CHI, ACT_MINKAN, ACT_ADD_KAN, ACT_PON)` 裁决，**吃竟然优先于碰和明杠**；且他家弃牌响应阶段不应出现 `ACT_ADD_KAN`（加杠只能发生在自己摸牌阶段）。

5. **动作编码与观测对副露不友好**
   - `train_hanchan.py:86-87`：`slot_of(ACT_CHI)` 用 `SLOT_CHI_BASE + (action.tile % 3)`，这是按绝对牌号取模哈希，而不是按「被吃牌在顺子中的相对位置（左吃/中吃/右吃）」，网络难以跨牌号泛化吃牌语义。
   - `mahjong/obs2.py:108-111, 150-151`：响应阶段只编码了 `last_discard`（第 17 平面）和 `last_discard_player == p_idx`（永远为 0），**未编码是谁打的弃牌（上/对/下家）**，也未显式编码候选副露涉及的牌面。

---

## 2. 不懂弃胡（Betaori / 防守）：观测致盲 + 奖励反激励

### 2.1 观测层对对手立直几乎「失明」
- `mahjong/obs2.py:116-119, 144`：
  - 第 19 平面 `if p.riichi: planes[19, :] = 1.0` —— **仅自家立直**。
  - 标量 `s[15] = 1.0 if p.riichi else 0.0` —— **仅自家立直**。
- 整个 `obs2` 的 22 个平面 + 26 个标量里，**完全没有 3 家对手各自的立直标志**、**立直宣言巡目**、以及**针对每个立直家的现物（安全牌）平面**（包括该立直家自己的牌河 + 立直宣告后所有玩家打出的安全通牌）。
- `mahjong/situation.py:95` 虽然在 12 维局势向量里加了 `riichi_count / 4.0`，但网络只知道「场上有人立直」，既不知道**哪家**立直，也无法直接查表哪些牌是现物，只能从 `event_stream`（如果启用了 `DualStreamMahjongNet`）里隐式推断。

### 2.2 训练信号与数据都在阻止 AI 学会弃和
1. **教师数据零防守**：`train_teacher_bc.py` 依赖 `GreedyTenpai`（`mahjong/policies.py:61-160`），其逻辑是无脑最小化向听数 + 有立直就立直，遇到对手立直从不兜牌/弃和。拿它做 BC 初始化，策略天然只有进攻没有防守。
2. **向听塑形直接惩罚弃和**：`train_ppo.py:199` 及多轮 RL 实验开启了 `shanten_weight = 0.3`，每步给 `0.3 * (sb - sa)`。在日麻中，面对立直切现物弃和必然拆面子/搭子（向听数上升），`sb - sa < 0` 立刻给予显著负奖励，把任何弃和探索扼杀在摇篮里。
3. **缺少半庄顺位博弈目标**：`mahjong/scenarios.py:64-65` 只以单局 `points_delta / 10000` 为主奖励（且 `points_delta` 未扣除自己立直时交出的 1000 点供托，导致立直成本在单局奖励中失真）。`mahjong/hanchan.py:145` 虽然定义了半庄顺位点 `placement_bonus([30000, 10000, -10000, -30000])`，但 `train_hanchan.py` **从未将其回传到轨迹回报**。没有避四和保一位的顺位非线性效用，AI 无法理解「领先时宁可弃和也不对攻」。
4. **外挂危险张模型存在结构局限**：`danger_model.py:90-91` 明确跳过了所有副露对手（`if int(melds[k, t]) != 0: continue`），且仅用 108 维扁平 MLP，未利用弃牌顺序和立直宣言后通牌，导致实战验证（`rules/DANGER_VALIDATION_2026-11-05.md`）置信区间跨 0。

---

## 3. 不懂流听（荒牌流局听牌 / 形式听牌）：规则错位 + 路径阻断

### 3.1 规则与结算层 Bug
1. **流局罚符金额错误**（`mahjong/env.py:1135-1138`）：代码写成「每个不听者给每个听牌者 1000 点」。结果 2 听 2 不听时，每人变为 `±2000`（总流转 4000），而日麻标准规则总池恒为 3000，2 听 2 不听应为 `±1500`。
2. **庄家流局听牌不连庄 + 轮庄错误**（`mahjong/hanchan.py:96-107`）：
   - 流局分支无条件调用 `self._advance_round()`，庄家流局听牌不会连庄（在实战中，庄家终盘抢形式听牌保连庄是极重要的流听动机）。
   - 第 104 行和牌分支写成了 `self.dealer = result.winner`（子家和牌直接变庄家），而日麻规则应是庄家和牌连庄、子家和牌由下家轮庄 `(dealer + 1) % 4`。
3. **王牌（死牌山）与余牌倒计时脱节**（`mahjong/env.py:305, 312, 867` 与 `mahjong/obs2.py:145`）：
   - `_apply_waribashi` 设置了 `self.dead_wall_size = 14`，但 `_draw_tile` 只判断 `self.wall_idx >= len(self.wall)`，没有留出 14 张王牌。
   - 实际触发流局的是 `_advance` 中的 `self.turn >= 72`，但副露不增加 `self.turn`，而 `obs2.py:145` 用 `(len(env.wall) - env.wall_idx) / 70.0` 告知网络剩余牌数。两者不同步，导致 AI 无法准确判断「还剩几巡流局」。
4. **开杠后听牌判定异常**（`mahjong/env.py:483-506`）：杠牌后 `full_counts()` 总张数为 14（因杠占 4 张），`_is_tenpai` 看到 `total == 14` 会误删 1 张牌，导致杠后流局听牌判定错误。
5. **非标准主动流局动作干扰**（`mahjong/env.py:409-412, 676-679`）：引擎在听牌时生成了日麻中不存在的 `ACT_RYUUKYOKU` / `ACT_NAGASHI` 主动流局动作，虽在训练脚本里被过滤，但属于规则层脏设计。

### 3.2 策略与训练层阻断
- **终盘抢形式听牌（keiten）依赖副露**：实战中大量流听是通过最后 2–3 巡吃/碰凑形式听牌实现的。因为第 1 节的副露锁死 Bug，模型根本不敢在终盘副露。
- **`deserves_give_up` 标签误导**：`mahjong/situation.py:134-158` 在 `turn_ratio > 0.66` 且未听牌且落后时直接打 `give_up` 标签，与「终盘在安全前提下抢形式听牌拿 +1500/+3000」的方向相反。
- **信噪比过低**：单局奖励 `delta / 10000` 下，流听盈亏仅 `±0.1 ~ ±0.3`，在纯蒙特卡洛回报的高方差下几乎无法产生有效梯度。

---

## 4. train 不出来（RL 退化与收敛失败）：底层与算法双重瓶颈

1. **规则引擎与参考算分器处于「半切换」状态**
   - `mahjong/refengine.py` 引入了开源 `MahjongRepository/mahjong`，但 `env.py` 的 `_best_score`（第 523–558 行）**仍然调用自研的 `hand.py` + `scoring.py` + `yaku.py`**，只在 `_can_tsumo`/`_can_ron` 前置调用了一次 `REF.can_win_with`（且未传 `melds`/`seat_wind`/`round_wind`/`is_riichi`）。
   - 同时 `refengine.py:42-44` 的 `to136(tile34) = tile34 * 4` 把同类 4 张牌全部映射成同一个 136 编号（如 `[0, 0, 0]`），直接用于带副露/赤宝牌的 `HandCalculator` 也会触发异常。
   - **后果**：环境本身的奖励函数就是错的，RL 越优化越偏离真实麻将策略。

2. **RL 算法实现偏差与极高方差**
   - `train_ppo.py` 虽以 PPO 命名，但实际没有 Critic、没有 Ratio Clip、没有 GAE，只是带 `shanten_weight=0.3` 的 vanilla REINFORCE（`adv = (rt - rt.mean()) / rt.std()`）。
   - `train_hanchan.py:547-555, 597` 虽补上了 PPO clip、Critic warmup 和 KL 锚定，但优势估计仍是**纯蒙特卡洛回报 `rt_chunk - values`**（无 GAE($\lambda$)），每次迭代仅采样 16 个半庄。在 4 人隐藏信息博弈中，MC 优势方差极大，导致 `rules/PPO_2026-11-05.md` 中从 BC 微调后和牌率从 33.75% 一路退化到 15.00%。
   - 未将半庄终局顺位点（Uma）归因到各局，只优化单局 `points_delta`（且漏算立直棒 1000 点成本）。

3. **吞吐量不足以支撑从零 RL**
   - `logs_train_v3.txt` 显示单机吞吐仅 `0.4 g/s`（1575 秒跑 612 局，0% 和牌）。立直麻将从零 RL 通常需要数千万局量级（纯 Python 引擎 + 逐巡规则计算无法在合理时间内提供足够样本）。

4. **默认网络结构瓶颈与特征割裂**
   - `mahjong/net.py:54, 92` 默认的 `MahjongNet` 用 1D 卷积横跨 34 维（`9m` 与 `1p`、`9s` 与 `东` 跨花色边界卷积污染），且 `per_tile_head=False` 时经 `AdaptiveAvgPool1d(4)` 压扁了单牌空间分辨率。
   - 你后续设计的 `DualStreamMahjongNet`（分花色共享卷积 + 字牌独立 MLP + 事件流 Cross-Attention + 逐牌切牌头）架构方向正确，但受限于上述引擎 Bug 与 `obs2` 缺失对手立直/现物特征，尚未发挥真正潜力。
   - 仓库里已有天凤凤桌牌谱解码器（`houou_data.py`，2.5M+ 局人类高手对局，天然包含高质量副露、弃和、流听），但主线训练仍反复在 `GreedyTenpai` 合成数据上打转，导致上限被规则教师锁死。
