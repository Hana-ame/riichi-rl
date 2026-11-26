# dest_prob_v3 防守权重分析（2026-11-26）

## 结论

**dest_prob_v3 默认参数（defense_weight=1.0）最优。** 提高防守权重降低放铳率，但牺牲进攻更多，Elo 反而下降。**进攻+防守平衡已是局部最优**。

## 证据

### dest_prob_v3 vs pure_shanten 800 局头对头

| bot | 平均pt | 1位% | 4位% | 和牌率 | 放铳率 |
|---|---|---|---|---|---|
| **dest_prob_v3** | **+12.57** | **49.2%** | 15.9% | 34.9% | **2.6%** |
| pure_shanten | +3.23 | 33.0% | 23.5% | 15.0% | 9.6% |

差值 **+9.34 pt/局**，1位率 49% vs 33%，放铳 2.6% vs 9.6%（**3.7 倍低**）。**dest_prob_v3 完全压倒 pure_shanten**。

**注**：之前 400 局 arena（seed=100）中 pure_shanten 第 1（1677.0）、dest_prob_v3 第 2（1665.2）是**arena 选员随机性噪声**，800 局直接头对头确认 dest_prob_v3 真的更强。

### dest_prob_v3 防守权重扫描（400 局 arena，选员制）

| 参赛者 | defense_weight | Elo | 和牌率 | 放铳率 |
|---|---|---|---|---|
| **dest_prob_v3** | **1.0（默认）** | **1665.2** | **33.3%** | 7.4% |
| dest_prob_v3_def15 | 1.5 | 1633.9 | 29.1% | **4.9%** |
| dest_prob_v3_def20 | 2.0 | 1559.5 | 22.8% | 7.6% |

**结论**：
- def15（1.5）：放铳降到 4.9%（最低），但和牌率从 33.3% 掉到 29.1%，Elo 掉 31 分
- def20（2.0）：防守过强，放铳反升 7.6%（因为过度回避进攻导致被动），Elo 掉 106 分
- **默认（1.0）是攻防平衡的局部最优**

### dest_prob_v3 多次 arena 表现

| arena | 局数 | seed | dest_prob_v3 Elo | 排名 |
|---|---|---|---|---|
| ELO_bc_destprob | 200 | ? | 1606.3 | 3 |
| ELO_bc_pureshanten_final | 400 | 0 | 1655.6 | 1 |
| ELO_dest_prob_param | 400 | 100 | 1665.2 | 2 |
| ELO_all_with_destprob_600 | 600 | 0 | **1740.8** | **1** |
| H2H vs pure_shanten | 800 | 1000 | 平均pt +12.57 | 压倒 |

**600 局最强证据**：dest_prob_v3 **1740.8 第 1**（天凤 pt +11.2/局，放铳 4.7%）。

## 天花板

**dest_prob_v3 天凤 1740.8 是当前天花板**，靠 4.7% 低放铳 + 33% 高和牌率的攻防平衡。ML（eff_wuke_d2 1577.4、BC 模仿 1460–1461）均远低于此。

## 后续方向

1. **直接用 dest_prob_v3 打排位**——已是天花板，无需 ML
2. **微调 defense_weight 1.1–1.3**——参数扫描脚本已建（arena_dest_prob_def_sweep.py），但需 300+ 局才能收敛，性价比低
3. **RL 突破教师**——BC 只能模仿，RL 自博弈可能突破 1740.8，但训练成本极高

## 文件

- `arena_dest_prob_param.py`：dest_prob_v3 参数变体 arena（注册 def15/def20/nocall）
- `arena_dest_prob_def_sweep.py`：defense_weight 扫描头对头（4 bot 同桌）
- `arena_h2h_dest_vs_shanten.py`：dest_prob_v3 vs pure_shanten 头对头
- `rules/H2H_dest_vs_shanten.md`：800 局头对头结果
- `rules/ELO_dest_prob_param.md`：400 局参数变体 arena 结果
- `rules/ELO_FINAL_2026-11-26.md`：排位训练最终结论
