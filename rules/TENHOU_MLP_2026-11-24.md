# Tenhou 离线 BC：管道跑通，MLP val_acc 0.5907（2026-11-24）

## 数据来源
HuggingFace `1nfrared218/tenhou-mjai-riichi-training`（CC-BY-4.0，鳳凰卓 2009-2026）。
- 下载脚本 `data/dl_houou_ext.sh` → 修到 `data/houou/dl_small.sh`（2024 chunk 00008–17）
- 关键修复 1：`"${y}_${c}.tar"` 被 bash 解析成变量名 `y_` → 文件名变 `_00016.tar`；改用 `"$y"_"$c".tar`
- 关键修复 2：加 HF API size 校验（远端 23M，本地截断到 1-9M 也要 size 完全匹配才 move）
- 最终：6 shard 2024_00009/10/11/13/14/16（缺 08/12/15 因三次重试都不匹配被丢弃）

## 特征（`tenhou_offline_bc.py`，560 维）
```
0-33    自己手牌（34）
34-67   自己弃牌（34）
68-203  四家弃牌（4×34=136）
204-339 四家副露（4×34，houou_data.melds_by 是 [(t1,t2,t3),...]）
340-509 dora 指示牌 5×34
510-513 四家立直
514-516 kyoku/honba/tiles_left
517-520 手牌/14, 副露/4, 轮次/20, 门清
521-523 trigger_type one-hot
524-558 trigger_tile 有效位 + one-hot（34）
```

## 关键 bug（每个都让 0 样本输出）
1. `f` 用 `dtype=np.uint8`：索引 `340+i*34+base` 当 `base=255`（ind=0 下溢）溢出 → 改 `float32`
2. `d.dora_indicators[:5]` 是 numpy uint8：`ind-1` 在 ind=0 时下溢 → 加 `ind=int(ind)`
3. `d.melds_by` 是 `[(t1,t2,t3),...]` 元组列表，不是 flat tiles → 双层 for
4. 忘 import `decode_match`（只导了 `read_shard`） → NameError
5. `read_shard` 一次解整 shard 46 万 Decision → OOM；改流式逐 npz + 预分配

## 训练结果（`checkpoints/tenhou_mlp.pt`）
```
样本 200000  维度 560  6 shard（2024 年 6 天）
ActionMLP(560→512→512→256→47)
epoch 0 loss=2.3628 val_acc=0.4881
epoch 4 loss=1.2427 val_acc=0.5836
epoch 7 loss=1.0339 val_acc=0.5907  ← best
```

## 判读（不修饰）
- 管道全通：HF 下载 → tar 解码 → 47 槽标签 → MLP → 权重保存
- val_acc 0.59 远低于自对弈 BC 的 0.98，**但两者不可直接比**：
  自对弈 val 集是同一策略的轨迹（分布窄），Tenhou val 是人类鳳凰卓玩家
  （策略多样，标签熵高，人类打牌本身不是 argmax）
- 下一步：把 Tenhou MLP 作为**教师**蒸馏到 obs2(786)+conv 主模型，看
  能否突破 BC 自对弈的天花板（14.69%@800 局池）

## 复现
```bash
PY=/home/lumin/miniconda3/bin/python
$PY data/houou/dl_small.sh  # 下载 2024 chunk 00008-17（约 230MB）
$PY tenhou_offline_bc.py --max-shards 6 --max-decisions 200000 \
    --epochs 8 --bs 512 --lr 1e-3 --out checkpoints/tenhou_mlp.pt
```

## 800 局门槛评测（`eval_selfplay_gate.py --tenhou-ckpt --deals 800 --seed0 96000`）

| 指标 | 读数 | 门槛 | 判定 |
|---|---|---|---|
| 每座位和牌 | **0.09%**（3/3200） | 15–25% | FAIL |
| 每座位放铳 | **0.06%**（2/3200） | 9–20% | FAIL |
| 流局 | **99.6%**（797/800） | — | — |
| 终局听牌 | 0.1% | — | — |
| 动作计数 | discard 57667 / riichi 0 / chi 0 / pon 149 / kan 89 / pass 172850 | — | pass 占 75% |

## 判读

MLP（560 粗特征）+ val_acc 0.59 **只能做分类，不能做对局决策**：
- 立直 0 次、自摸 1 次、荣和 2 次——几乎不会主动和牌；
- pass 占 75%（模型倾向选 PASS 槽 43，因为 Tenhou 数据里响应类 pass 多）；
- 99.6% 流局、终局听牌 0.1%——完全不懂得推进牌型；
- 根因：560 维粗特征丢失了**手牌结构/听牌/攻防**信息，MLP 容量也不够。

## 结论

Tenhou MLP 管道跑通但**不可用**。下一步：
- 方案 A：把 Tenhou 数据转 obs2(786)（需完整 MahjongEnv 回放，成本高）；
- 方案 B：Tenhou MLP 蒸馏到主模型（需对齐 obs2 特征，同样成本高）；
- 方案 C：用 Tenhou **统计特征**（弃牌分布/立直频率）辅助现有主模型，而非替换。

## Tenhou obs2(786) BC：val_acc 0.549，800 局门槛 2.19%/1.56%（仍然 FAIL）

把 Tenhou Decision 转成 obs2(786) 用主模型架构训练（`tenhou_to_obs2.py` → `data/tenhou_obs2.npz`，
`train_teacher_bc.py --channels 64 --blocks 4 --epochs 6` → `checkpoints/tenhou_bc.pt`）：

- 训练 val_acc 0.409→0.549（6 epochs）；叫牌 4718+509+7=5234 次（Tenhou 数据里有真实叫牌）；
- 800 局门槛（`eval_selfplay_gate.py --ckpt tenhou_bc --deals 800 --seed0 96000`）：

| 指标 | 读数 | 门槛 | 判定 |
|---|---|---|---|
| 每座位和牌 | **2.19%**（70/3200） | 15–25% | FAIL |
| 每座位放铳 | **1.56%**（50/3200） | 9–20% | FAIL |
| 流局 | 91.2% | — | — |
| 终局听牌 | 10.6% | — | — |
| 动作计数 | discard 60680 / riichi 190 / chi **4718** / pon 509 / kan 7 | — | — |

## 关键发现（对比 560 维 MLP）

| 模型 | val_acc | 和牌 | 放铳 | 叫牌次数 |
|---|---|---|---|---|
| 560 维 MLP（tenhou_mlp） | 0.59 | 0.09% | 0.06% | 149+89+0 |
| **obs2(786) conv（tenhou_bc）** | **0.549** | **2.19%** | **1.56%** | **4718+509+7** |

- 和牌/放铳/叫牌都**翻了几十倍**，但绝对值仍远低于门槛；
- 根因：**Tenhou 数据只有 20 万样本**（自对弈 BC 是 250 万），且标签是**人类玩家混合策略**
  （鳳凰卓有新手有高手），学生学到的是平均行为而非最优策略；
- 结论：Tenhou 数据可作**补充语料**，但不能单独当 BC 数据源；需要
  **自对弈 250 万 + Tenhou 20 万混合**再训练，看是否能在 14.69% 基础上加 buff。

## 混合训练最终结果（自对弈 249.7 万 + Tenhou 20 万，2026-11-24 10:22）

`train_teacher_bc.py --data data/mix_selfplus_tenhou.npz --init teacher47_eff_wuke_d2.pt
--epochs 3 --lr 2e-4` → `checkpoints/teacher47_wuke_tenhou.pt`（val_acc 0.947）

800 局门槛（`eval_selfplay_gate.py --ckpt teacher47_wuke_tenhou --deals 800 --seed0 96000`）：

| 指标 | 混合 Tenhou | eff_wuke_d2（对照） | 变化 |
|---|---|---|---|
| 每座位和牌 | **13.91%**（445/3200）CI[12.75,15.15] | 14.69%（470/3200） | **−0.78pt** |
| 每座位放铳 | 10.59%（339/3200）CI[9.57,11.71] | 11.69%（374） | −1.10pt |
| 流局 | 35.5% | 41.2% | −5.7pt |
| 终局听牌 | 53.8% | 54.2% | ≈ |
| 立直次数 | 1721 | 231 | +1490 |

## 判读（不修饰）

- 加 Tenhou 数据**没带来提升，反而降 0.78pt**：混合后 val_acc 从 0.982 降到 0.947，
  立直次数暴涨（231→1721）但和牌率下降——模型学到 Tenhou 玩家的"更爱立直"风格，
  但立直并不转化为和牌；
- **放铳 CI[9.57,11.71] 完全落在 9–20% 区间**（稳定 PASS）；
- 和牌 CI[12.75,15.15] 上界刚过 15，下界差 2.25pt → **未稳定达标**；
- 结论：**Tenhou 人类数据作补充语料是负收益**（在本引擎/本动作空间下）；
  当前 best 仍是 eff_wuke_d2（14.69%@800 / 合并 2800 局 14.50%）。
- 下一可试方向（按预期收益排序）：
  1. eff_wuke_d2 继续跑第 3 轮 DAgger（前两轮 +0.31/+0.22pt）；
  2. 128ch/5blk 大模型从零训（之前 eff_big 失败是 val_acc 0.930 太低）；
  3. obs v3（+牌河顺序平面）从零训。
