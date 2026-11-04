# 模型版本登记表（2026-11-05）

> 自动扫描 `checkpoints/`：共 **91** 个模型文件、**11** 条血缘。

> 字段：版本（同血缘内按 mtime 排序）/ ckpt / 架构与参数量 / 训练元数据 / 赛评读数 / 状态（天梯闸门优先）。

## 纯 RL（riichi_net）（1 个）

| 版本 | ckpt | 架构 | 参数量 | 训练元数据 | 赛评读数 | 状态 |
|---|---|---|---|---|---|---|
| V1 | `riichi_net.pt` | convformer54 | 1,652,967 | - | -；ELO 1359.9（win 3.2%, 放铳 6.3%, pts -668.3） | 历史版本（无闸门记录） |

## A2C·47槽（早期 RL）（23 个）

| 版本 | ckpt | 架构 | 参数量 | 训练元数据 | 赛评读数 | 状态 |
|---|---|---|---|---|---|---|
| V1 | `a2c_r9_it5.pt` | hanchan47 | 453,232 | iter=5 deals=80 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V2 | `a2c_r4_it10.pt` | hanchan47 | 453,232 | iter=10 deals=160 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V3 | `a2c_r9_it10.pt` | hanchan47 | 453,232 | iter=10 deals=160 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V4 | `a2c_r9_it15.pt` | hanchan47 | 453,232 | iter=15 deals=240 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V5 | `a2c_r4_it20.pt` | hanchan47 | 453,232 | iter=20 deals=320 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V6 | `a2c_r9_it20.pt` | hanchan47 | 453,232 | iter=20 deals=320 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V7 | `a2c_r9.pt` | hanchan47 | 453,232 | deals=400 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V8 | `a2c_r9_it25.pt` | hanchan47 | 453,232 | iter=25 deals=400 win_rate=0.0 channels=64 blocks=4 | - | 未通过闸门（不发布）（点差 +83 [-80,+247]） |
| V9 | `a2c_r4.pt` | hanchan47 | 453,232 | deals=480 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V10 | `a2c_r4_it30.pt` | hanchan47 | 453,232 | iter=30 deals=480 win_rate=0.0 channels=64 blocks=4 | - | 证据不足（HOLD）（点差 -186 [-441,+69]） |
| V11 | `a2c_r10_it5.pt` | hanchan47 | 453,232 | iter=5 deals=80 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V12 | `a2c_r1_it10.pt` | hanchan47 | 453,232 | iter=10 deals=160 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V13 | `a2c_r10_it10.pt` | hanchan47 | 453,232 | iter=10 deals=160 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V14 | `a2c_r10_it15.pt` | hanchan47 | 453,232 | iter=15 deals=240 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V15 | `a2c_r5_it10.pt` | hanchan47 | 453,232 | iter=10 deals=160 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V16 | `a2c_r1_it20.pt` | hanchan47 | 453,232 | iter=20 deals=320 win_rate=0.0 channels=64 blocks=4 | - | 证据不足（HOLD）（点差 +0 [-255,+255]） |
| V17 | `a2c_r10_it20.pt` | hanchan47 | 453,232 | iter=20 deals=320 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V18 | `a2c_r5_it20.pt` | hanchan47 | 453,232 | iter=20 deals=320 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V19 | `a2c_r10.pt` | hanchan47 | 453,232 | deals=400 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V20 | `a2c_r10_it25.pt` | hanchan47 | 453,232 | iter=25 deals=400 win_rate=0.0 channels=64 blocks=4 | -；ELO 1449.4（win 0.0%, 放铳 7.5%, pts -1208.3） | 证据不足（HOLD）（点差 +139 [-227,+505]） |
| V21 | `a2c_r1.pt` | hanchan47 | 453,232 | deals=480 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V22 | `a2c_r1_it30.pt` | hanchan47 | 453,232 | iter=30 deals=480 win_rate=0.0 channels=64 blocks=4 | - | 证据不足（HOLD）（点差 -108 [-478,+262]） |
| V23 | `a2c_r11_it5.pt` | hanchan47 | 453,232 | iter=5 deals=80 win_rate=0.0 channels=64 blocks=4 | - | 证据不足（HOLD）（点差 +267 [-38,+571]） |

## A2C·54槽（早期 RL）（7 个）

| 版本 | ckpt | 架构 | 参数量 | 训练元数据 | 赛评读数 | 状态 |
|---|---|---|---|---|---|---|
| V1 | `a2c_r8_it5.pt` | convformer54 | 455,031 | iter=5 deals=80 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V2 | `a2c_r2_it10.pt` | convformer54 | 455,031 | iter=10 deals=160 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V3 | `a2c_r8_it10.pt` | convformer54 | 455,031 | iter=10 deals=160 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V4 | `a2c_r2_it20.pt` | convformer54 | 455,031 | iter=20 deals=320 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V5 | `a2c_r2.pt` | convformer54 | 455,031 | deals=480 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V6 | `a2c_r2_it30.pt` | convformer54 | 455,031 | iter=30 deals=480 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V7 | `a2c_r3_it10.pt` | convformer54 | 455,031 | iter=10 deals=160 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |

## DAgger（12 个）

| 版本 | ckpt | 架构 | 参数量 | 训练元数据 | 赛评读数 | 状态 |
|---|---|---|---|---|---|---|
| V1 | `dagger_it10.pt` | hanchan47 | 453,232 | iter=10 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V2 | `dagger_it20.pt` | hanchan47 | 453,232 | iter=20 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V3 | `dagger_it30.pt` | hanchan47 | 453,232 | iter=30 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V4 | `dagger_it40.pt` | hanchan47 | 453,232 | iter=40 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V5 | `dagger_it50.pt` | hanchan47 | 453,232 | iter=50 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V6 | `dagger_it60.pt` | hanchan47 | 453,232 | iter=60 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V7 | `dagger_it70.pt` | hanchan47 | 453,232 | iter=70 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V8 | `dagger_it80.pt` | hanchan47 | 453,232 | iter=80 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V9 | `dagger_it90.pt` | hanchan47 | 453,232 | iter=90 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V10 | `dagger_it100.pt` | hanchan47 | 453,232 | iter=100 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V11 | `dagger_it110.pt` | hanchan47 | 453,232 | iter=110 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V12 | `dagger_it120.pt` | hanchan47 | 453,232 | iter=120 channels=64 blocks=4 | -；ELO 1384.5（win 0.0%, 放铳 0.0%, pts -786.2） | 历史版本（无闸门记录） |

## Hanchan HC（37 个）

| 版本 | ckpt | 架构 | 参数量 | 训练元数据 | 赛评读数 | 状态 |
|---|---|---|---|---|---|---|
| V1 | `hc_r9_it5.pt` | convformer54 | 455,031 | iter=5 deals=160 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V2 | `hc_r9_it10.pt` | convformer54 | 455,031 | iter=10 deals=160 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V3 | `hc_r12_it5.pt` | hanchan47 | 453,232 | iter=5 deals=80 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V4 | `hc_r13_it5.pt` | hanchan47 | 453,232 | iter=5 deals=80 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V5 | `hc_r10_it5.pt` | hanchan47 | 453,232 | iter=5 deals=160 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V6 | `hc_r12_it10.pt` | hanchan47 | 453,232 | iter=10 deals=160 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V7 | `hc_r13_it10.pt` | hanchan47 | 453,232 | iter=10 deals=160 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V8 | `hc_r12_it15.pt` | hanchan47 | 453,232 | iter=15 deals=240 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V9 | `hc_r13_it15.pt` | hanchan47 | 453,232 | iter=15 deals=240 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V10 | `hc_r10_it10.pt` | hanchan47 | 453,232 | iter=10 deals=320 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V11 | `hc_r12_it20.pt` | hanchan47 | 453,232 | iter=20 deals=320 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V12 | `hc_r13_it20.pt` | hanchan47 | 453,232 | iter=20 deals=320 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V13 | `hc_r10_it15.pt` | hanchan47 | 453,232 | iter=15 deals=480 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V14 | `hc_r12_it25.pt` | hanchan47 | 453,232 | iter=25 deals=400 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V15 | `hc_r13_it25.pt` | hanchan47 | 453,232 | iter=25 deals=400 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V16 | `hc_r10.pt` | hanchan47 | 453,232 | deals=480 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V17 | `hc_r13_it30.pt` | hanchan47 | 453,232 | iter=30 deals=480 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V18 | `hc_r12_it30.pt` | hanchan47 | 453,232 | iter=30 deals=480 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V19 | `hc_r10_it20.pt` | hanchan47 | 453,232 | iter=20 deals=640 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V20 | `hc_r11_it5.pt` | hanchan47 | 453,232 | iter=5 deals=80 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V21 | `hc_r12_it35.pt` | hanchan47 | 453,232 | iter=35 deals=560 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V22 | `hc_r13_it35.pt` | hanchan47 | 453,232 | iter=35 deals=560 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V23 | `hc_r11_it10.pt` | hanchan47 | 453,232 | iter=10 deals=160 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V24 | `hc_r12_it40.pt` | hanchan47 | 453,232 | iter=40 deals=640 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V25 | `hc_r13_it40.pt` | hanchan47 | 453,232 | iter=40 deals=640 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V26 | `hc_r10_it25.pt` | hanchan47 | 453,232 | iter=25 deals=800 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V27 | `hc_r11_it15.pt` | hanchan47 | 453,232 | iter=15 deals=240 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V28 | `hc_r13_it45.pt` | hanchan47 | 453,232 | iter=45 deals=720 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V29 | `hc_r12_it45.pt` | hanchan47 | 453,232 | iter=45 deals=720 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V30 | `hc_r10_it30.pt` | hanchan47 | 453,232 | iter=30 deals=960 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V31 | `hc_r11_it20.pt` | hanchan47 | 453,232 | iter=20 deals=320 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V32 | `hc_r12_it50.pt` | hanchan47 | 453,232 | iter=50 deals=800 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V33 | `hc_r13_it50.pt` | hanchan47 | 453,232 | iter=50 deals=800 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V34 | `hc_r11_it25.pt` | hanchan47 | 453,232 | iter=25 deals=400 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V35 | `hc_r12_it55.pt` | hanchan47 | 453,232 | iter=55 deals=880 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |
| V36 | `hc_r13_it55.pt` | hanchan47 | 453,232 | iter=55 deals=880 win_rate=0.0 channels=64 blocks=4 | -；ELO 1496.8（win 0.0%, 放铳 3.7%, pts -1035.8） | 历史版本（无闸门记录） |
| V37 | `hc_r10_it35.pt` | hanchan47 | 453,232 | iter=35 deals=1120 win_rate=0.0 channels=64 blocks=4 | - | 历史版本（无闸门记录） |

## BC 人类牌谱教师（2 个）

| 版本 | ckpt | 架构 | 参数量 | 训练元数据 | 赛评读数 | 状态 |
|---|---|---|---|---|---|---|
| V1 | `bc_v1.pt` | bc243 | 717,076 | epoch=4 val_acc=0.5921666666666666 channels=96 blocks=4 | - | 历史版本（无闸门记录） |
| V2 | `bc_v1_last.pt` | bc243 | 717,076 | epoch=4 val_acc=0.5921666666666666 channels=96 blocks=4 | -；ELO 1666.3（win 16.4%, 放铳 2.7%, pts +1243.8） | 在用（和牌率达标） |

## 蒸馏（from_games）（2 个）

| 版本 | ckpt | 架构 | 参数量 | 训练元数据 | 赛评读数 | 状态 |
|---|---|---|---|---|---|---|
| V1 | `from_games_v1.pt` | convformer54 | 1,652,967 | - | -；ELO 1450.5（win 0.0%, 放铳 1.5%, pts -605.0） | 历史版本（无闸门记录） |
| V2 | `from_games_v2.pt` | convformer54 | 1,652,967 | - | -；ELO 1692.8（win 0.4%, 放铳 3.5%, pts -179.3） | 历史版本（无闸门记录） |

## 教师 BC 固定槽（teacher47）（3 个）

| 版本 | ckpt | 架构 | 参数量 | 训练元数据 | 赛评读数 | 状态 |
|---|---|---|---|---|---|---|
| V1 | `teacher47_v1.pt` | hanchan47 | 453,232 | iter=6 deals=-1 win_rate=0.0 channels=64 blocks=4 | 自对弈 5.0%（40 局） | 历史版本（无闸门记录） |
| V2 | `teacher47_v2.pt` | hanchan47 | 453,232 | iter=8 deals=-1 win_rate=0.0 channels=64 blocks=4 | 自对弈 15.0%（80 局） | 证据不足（HOLD）（点差 -135 [-806,+535]） |
| V3 | `teacher47_v3.pt` | hanchan47 | 453,232 | iter=6 deals=-1 win_rate=0.0 channels=64 blocks=4 | 自对弈 17.5%（80 局）；ELO 1435.8（win 5.1%, 放铳 5.1%, pts -179.2） | 通过闸门（可发布）（点差 +497 [+7,+987]） |

## 风格微调（style）（2 个）

| 版本 | ckpt | 架构 | 参数量 | 训练元数据 | 赛评读数 | 状态 |
|---|---|---|---|---|---|---|
| V1 | `style_rl_v1.pt` | hanchan47 | 453,232 | deals=192 win_rate=1.0416666666666667 channels=64 blocks=4 | 自对弈 7.5%（40 局，退化） | 未通过闸门（不发布）（点差 -796 [-1229,-362]） |
| V2 | `style_rl_v1_it5.pt` | hanchan47 | 453,232 | iter=5 deals=160 win_rate=1.25 channels=64 blocks=4 | - | 历史版本（无闸门记录） |

## 对手手牌推断（1 个）

| 版本 | ckpt | 架构 | 参数量 | 训练元数据 | 赛评读数 | 状态 |
|---|---|---|---|---|---|---|
| V1 | `opponent_model.pt` | opponent_model | 65,186 | samples=155289 val_loss=0.22631923854351044 | - | 历史版本（无闸门记录） |

## 和牌预测器（1 个）

| 版本 | ckpt | 架构 | 参数量 | 训练元数据 | 赛评读数 | 状态 |
|---|---|---|---|---|---|---|
| V1 | `hand_predictor.pt` | hand_predictor | 43,779 | samples=51666 base_win=0.00946463830769062 | - | 历史版本（无闸门记录） |

## 说明

- 版本号只在**同一血缘内部**排序，跨血缘不可比。
- 状态优先级：天梯闸门（PASS/FAIL/HOLD） > ELO 和牌率是否达标（≥15%）> 历史版本。
- 天梯闸门来自 `version_ladder.py` 的规则：相邻版本同局净点差 95% CI 下界 >0 且 Elo 未显著下滑才 PASS。
- 生成时间：2026-11-05 04:26；权重文件本身不入库（`.gitignore` 排除 `*.pt`）。
