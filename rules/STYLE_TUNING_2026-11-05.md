# 断幺/平和等牌风收进「一个模型」调参（2026-11-05）

> 回应：「断幺人、平胡人要弄到一个模型里面调参」。
> 做法：不再为牌风拆不同模型/人——在**同一个 47 槽模型**的训练 reward 里加
> 可调的风格权重（`--aux-weights tanyao=…,pinfu=…`），用参数控制“多想做断幺/
> 平和”。通路上线；第一轮短程微调**掉和牌率**（15%→7.5%），参数还没调好，
> 如实记录与下一步。

## 1. 实现（同一模型、同一 ckpt、只改权重）

- **风格任务**（`mahjong/aux_tasks.py`）：
  - `tanyao`（和牌且役含断幺九）+300
  - `pinfu`（和牌且役含平和）+300
  - `bonus_of(done, weights)` 支持逐任务倍率；`deal_bonus(..., weights=)`
- **训练调参旋钮**（`train_hanchan.py`）：
  - `--aux-bonus-weight`（任务奖励进入 reward 的总权重）
  - `--aux-weights "tanyao=3,pinfu=3"`（各风格倍率）
  - `--init checkpoints/teacher47_v3.pt`（从同一模型继续训练，不换模型）
- **评估**（`eval_hanchan.py`）：`run()` 现在同时打印 **断幺率/平和率**
  （胜局中出现该役的比例），用于看“风格权重是否真的改变牌风”。

## 2. 用法示例（同一模型，不同参数 = 不同牌风）

```bash
cd /mnt/e/riichi-rl
PY=/mnt/d/WorkPlace/my_rl_project/venv/bin/python

# 更爱断幺/平和：
$PY train_hanchan.py --init checkpoints/teacher47_v3.pt \
    --iters 12 --hanchans 4 --aux-bonus-weight 0.5 \
    --aux-weights "tanyao=3,pinfu=3" --shanten-weight 0.3 --temperature 0.7 \
    --out checkpoints/style_rl_v1.pt

# 只调一路（例如只爱断幺）就换权重，模型文件/结构完全不变：
--aux-weights "tanyao=3"
```

## 3. 第一轮读数（同一批种子 40 局）

| 模型 | 和牌率 | 断幺率 | 平和率 |
|---|---|---|---|
| 教师向听贪心（baseline） | 35.0% | 0.0% | 21.4% |
| teacher47_v3（加权 BC，微调前） | 15.0% | 0.0% | 0.0% |
| style_rl_v1（权重 tanyao=3/pinfu=3，微调 6 iter） | **7.5%** | 0.0% | 0.0% |

- 微调 6 iter（192 局）后训练内胜率从 3.1% 掉到 1.0%，**策略退化**；
  40 局里只胜 3 局，断幺/平和命中率样本太小看不出偏移（教师模型本身
  和牌也偏门前清/立直，断幺/平和为 0）。
- 结论：**通路已就绪（一个模型 + 风格权重可调）**，但「短程 RL 从 BC 初始化
  直接改 reward」会破坏原策略——这正是需要继续调的参数。

## 4. 下一步调参计划（按优先级）

1. 更稳的微调：lr 3e-5~5e-5、iters 50+、entropy 0.02，加 BC 锚正则
   （对 teacher47 的动作做 KL 约束），先保住 15% 和牌率再谈风格偏移。
2. 风格监督更直接：`gen_teacher_data.py` 记录终局 yaku，做**带权 BC**
   （断幺/和平和了的决策样本加权），比 RL 调 reward 更稳。
3. 评估要够样本：当前 40 局里模型只胜 3–6 局，断幺/平和率噪声极大；
   至少 300 局或按“胜局数”给置信区间。
4. 风格权重不要在推理时改结构——同一 ckpt 固定权重训练一版；想切风格就
   用不同权重再训一版（或以后做成 style 向量输入，属于下一阶段）。

## 5. 复现

```bash
$PY -m tests.run_all                      # 83/83（新增风格任务测试 3 例）
$PY eval_hanchan.py --ckpt checkpoints/teacher47_v3.pt --hanchans 5
$PY eval_hanchan.py --ckpt checkpoints/style_rl_v1.pt --hanchans 5
```
