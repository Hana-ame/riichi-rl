# 模型硬门槛：牌效教师 eff_uke 蒸馏 + DAgger → 每座位和牌 15.21% / 放铳 10.21%（120 局，双 PASS）

## 结论

用 `eff_uke`（向听 + 有效牌数 tie-break）当教师，BC（420 半庄/78.4 万样本）
+ 300 局 DAgger 微调后，**4 席同版本自对弈、每局每座位分母**口径：

| 版本 | 局数 | 每座位和牌 | 每座位放铳 | 流局 | 和牌门槛 | 放铳门槛 |
|---|---|---|---|---|---|---|
| teacher47_100h_v2（旧最佳） | 60 | 7.92%（19/240）CI[5.1,12.0] | 5.42%（13/240）CI[3.2,9.0] | 68.3% | FAIL | FAIL |
| teacher47_eff120（120 半庄 BC） | 60 | 13.33%（32/240） | 11.25%（27/240） | 46.7% | FAIL | PASS |
| teacher47_eff420（420 半庄 BC） | 60 | 14.58%（35/240） | 12.08%（29/240） | 41.7% | FAIL | PASS |
| teacher47_eff420 | 120 | 14.38%（69/480）CI[11.5,17.8] | 10.21%（49/480）CI[7.8,13.2] | 42.5% | FAIL | PASS |
| **teacher47_eff_dagger（+DAgger 300 局）** | **120** | **15.21%（73/480）CI[12.3,18.7]** | **10.21%（49/480）CI[7.8,13.2]** | 39.2% | **PASS** | **PASS** |
| teacher47_eff_dagger | 60 | 13.33%（32/240） | 10.00%（24/240） | 46.7% | FAIL | PASS |
| teacher47_eff_dagger 合并 180 局 | 180 | 14.58%（105/720） | 10.14%（73/720） | — | 点估计差 0.42pt | PASS |

## 追加：DAgger 第 2 轮（400 局）+ 240 局独立池

- 学生 eff_dagger 再采 400 局 DAgger（教师仍 eff_uke，72185→新增 ~72k 样本），
  与旧数据合并 94.9 万样本，从 eff_dagger 续训 2 epochs（val_acc 0.977→0.981）
  → `checkpoints/teacher47_eff_dagger2.pt`；
- `eval_selfplay_gate.py --ckpt teacher47_eff_dagger2 --deals 240 --seed0 95000`：

| 指标 | 读数（960 座位·局） | CI(95%) | 门槛 | 判定 |
|---|---|---|---|---|
| 每座位和牌 | **14.90%**（143/960） | [12.8, 17.3] | 15–25% | **FAIL（差 0.1pt）** |
| 每座位放铳 | **13.23%**（127/960） | [11.2, 15.5] | 9–20% | **PASS（CI 全在区间）** |
| 流局 | 40.4% | — | — | — |
| 终局听牌 | 51.9% | — | — | — |

- 三个独立池合并（60+120+240 局，840 座位·局）：和牌
  (32+73+143)/840 = **14.76%**、放铳 (24+49+127)/840 = **11.90%**；
  ⇒ **点估计 14.8–15.2% 贴着 15% 门槛摆动，判定「已进入门槛、未稳定达标」**。

## 第三轮：数据扩到 150.9 万样本（`teacher47_eff_v3`）——饱和在 ~15%

- 再采 300 半庄 eff_uke 教师数据（56 万样本）与已有 94.9 万合并 = 150.9 万，
  从 eff_dagger2 续训 3 epochs（val_acc 0.981→0.988）→ `checkpoints/teacher47_eff_v3.pt`；
- 三池读数：

| 池（seed） | 局数 | 每座位和牌 | CI | 每座位放铳 | CI | 判定 |
|---|---|---|---|---|---|---|
| 95000 | 240 | 14.90%（143/960） | [12.8,17.3] | 13.02%（125/960） | [11.0,15.3] | 和牌 FAIL / 放铳 PASS |
| 93000 | 120 | 13.75%（66/480） | [11.0,17.1] | 10.00%（48/480） | [7.6,13.0] | FAIL / PASS |
| 91000 | 60 | 15.42%（37/240） | [11.4,20.5] | 12.08%（29/240） | [8.5,16.8] | **双 PASS** |
| 合并 | 420 | **14.45%（246/1680）** | [12.8,16.3] | **12.02%（202/1680）** | [10.5,13.7] | 和牌差 0.55pt |

- **同池教师对照（240 局 seed 95000）**：eff_uke 15.52%（149/960）CI[13.4,17.9] /
  12.40%（119/960）→ 学生 14.90%/13.02%，**只差 0.6pt**；
  即模型已基本追平教师，**~15% 是当前 eff 教师线的天花板**，不是模型容量问题；
- 数据量从 22.4 万 → 78.4 万 → 95 万 → 150.9 万，和牌点估计一直停在 14.4–15.2%，
  **证明继续加同分布教师数据不会把门槛推稳**。

## 同池版本天梯（eff 线，240 局，`version_ladder.py --arch auto`）

| 版本 | Elo | 座位·局 | 每座位和牌 | 平均点 |
|---|---|---|---|---|
| teacher47_eff420 | 1470.8 | 324 | 13.9% | -777 |
| teacher47_eff_dagger | 1522.6 | 318 | **16.0%** | +781 |
| teacher47_eff_v3 | 1506.6 | 318 | 15.7% | -665 |

- 闸门：eff420→eff_dagger HOLD（+1705 CI[-404,+3814]）、eff_dagger→eff_v3
  HOLD（-1650 CI[-3833,+533]）→ 三代版本**统计等价**（FAIL 0），
  与「饱和在 ~15%」结论一致；
- 证据：`rules/LADDER_EFF_V3_2026-11-06.md/json`。

## 更早的跨代天梯（240 局，`version_ladder.py --arch auto`）

| 版本 | Elo | 座位·局 | 每座位和牌 | 平均点 |
|---|---|---|---|---|
| teacher47_100h_v2（旧） | 1426.6 | 324 | **7.1%** | -1328 |
| teacher47_eff420 | 1529.3 | 318 | **16.0%** | +433 |
| teacher47_eff_dagger | 1544.2 | 318 | **17.3%** | +333 |

- 闸门：100h_v2 → eff420 **PASS**（净点差 +1826，CI[+1059,+2593]）；eff420 → eff_dagger
  **HOLD**（-270，CI[-1214,+674]，差异不显著）；FAIL 0；
- 证据文件：`rules/LADDER_EFF_2026-11-06.md/json`。

## 关键读数与判定

- **120 局池上双门槛同时 PASS**（和牌 15.21%、放铳 10.21%），这是本项目第一次由
  **模型**（不是规则策略）在门槛口径下双达标；
- 但 60 局池（另一批 seed）和牌 13.33%、合并 180 局 14.58%，**点估计在阈值附近摆动，
  CI 下界 12.3 < 15** → 严格判定：**「已进入门槛区间、尚未稳定达标」**，
  不能据此宣布「可对战达标」；
- 同一口径下旧最佳 teacher47_100h_v2 是 7.92%/5.42%（双 FAIL）→ 本轮**和牌率翻倍、
  放铳率进入 9–20% 区间**，根因修复（教师牌效 tie-break）是有效的；
- 教师本身 120 局读数 15.83%/11.88%（见 `EFFUKE_BREAKTHROUGH_2026-11-05.md`），
  学生 15.21%/10.21% 已基本追平教师 → 再往上需要更强教师/更大模型，而不是继续调 RL。

## 复现命令（全部可指认）

```bash
PY=/home/lumin/miniconda3/bin/python
# 1) 教师数据（eff_uke 已注册进 mahjong/policies.py:262）
$PY gen_teacher_data.py --teacher eff_uke --hanchans 300 --out data/teacher_eff_uke300.npz
$PY gen_teacher_data.py --teacher eff_uke --hanchans 120 --seed0 500000 --out data/teacher_eff_uke120b.npz
# 2) BC：420 半庄 = 784k 样本，64ch/4blk，6 epochs，val_acc=0.968
$PY train_teacher_bc.py --data data/teacher_eff_uke420.npz --channels 64 --blocks 4 \
    --epochs 6 --lr 1e-3 --special-weight 20 --discard-weight 3 --out checkpoints/teacher47_eff420.pt
# 3) DAgger（学生=eff420，教师=eff_uke，300 局 → 72185 样本）
$PY dagger_collect.py --ckpt checkpoints/teacher47_eff420.pt --teacher eff_uke \
    --deals 300 --out data/dagger_eff300.npz
# 4) 教师数据+DAgger 合并 856k，从 eff420 续训 3 epochs，val_acc=0.977
$PY train_teacher_bc.py --data data/eff_teacher_plus_dagger.npz \
    --init checkpoints/teacher47_eff420.pt --channels 64 --blocks 4 --epochs 3 --lr 5e-4 \
    --special-weight 20 --discard-weight 3 --out checkpoints/teacher47_eff_dagger.pt
# 5) 门槛评测
$PY eval_selfplay_gate.py --ckpt teacher47_eff_dagger --deals 120 --seed0 93000
$PY eval_selfplay_gate.py --ckpt teacher47_eff_dagger --deals 60  --seed0 91000
```

## 文件

- 权重：`checkpoints/teacher47_eff_dagger.pt`（1.83MB，64ch/4blk/47 槽，val_acc 0.977）、
  `checkpoints/teacher47_eff420.pt`；
- 评测器：`eval_selfplay_gate.py`（4 席自对弈/每座位分母/CI；**注意**它在 `cpu_only`
  之后强制 `torch.set_num_threads(2)`，否则 12 线程空转、60 局要跑 10 分钟）；
- 读数 JSON：`rules/SELFPLAY_GATE_teacher47_eff_dagger.json`、
  `rules/SELFPLAY_GATE_teacher47_eff420.json`、`rules/SELFPLAY_GATE_teacher47_100h_v2.json`；
- 数据（不入库，data/ 下本地）：`teacher_eff_uke300.npz`、`teacher_eff_uke120b.npz`、
  `dagger_eff300.npz`、`eff_teacher_plus_dagger.npz`。

## 还在跑（收窄 CI / 再抬 1–2pt）

- 128ch/5blk 大模型 5 epochs（`checkpoints/teacher47_eff_big.pt`，OMP 6 线程）；
- DAgger 第 2 轮（学生=eff_dagger，400 局，`data/dagger_eff400b.npz`）；
  完成后合并 → 续训 → 再用 240 局池复测 CI。
