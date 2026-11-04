# 固定槽位 + 特殊动作加权 BC：让模型自己会和牌（2026-11-05 实验）

> 目标（第二轮审阅定的 P1）：**不用规则接管**，让神经网络自己立直/和牌。
> 结果：纯模型和牌率 **2.5% → 17.5%**（80 局同种子对照），模型自选立直/和牌
> 的命中率 **100%**；仍未达到 >20% 的目标，原因与下一步见 §5。
> 产出（磁盘，不入 git）：`checkpoints/teacher47_v1/v2/v3.pt`、`data/teacher47.npz`。

## 1. 为什么另起一条线（不是继续训 54 槽）

54 槽线 `train_reinforce.slot_of` 是哈希 `34+((type*7+tile)%20)`，**非单射**：
40 局 10,572 决策点里 66 个槽冲突（0.62%），全是 CHI/PON 撞 PASS(固定 50)，
`legal[slots.index(pick)]` 会把模型想叫的 CHI/PON 执行成 PASS，训练标签同样
被写成 50。半庄 47 槽线 `train_hanchan.slot_of` 用**固定特殊槽**
（TSUMO=34 / RON=35 / RIICHI=36 / PASS=43 / CHI=44..46），无此问题。

## 2. 方法

1. **教师数据**（`gen_teacher_data.py`）：GreedyTenpai×4 跑 **30 半庄（240 局）**，
   逐决策记录 `obs(786=774+12 局势) + touch(8) + 合法槽 mask(47) + 槽标签`。
   共 **62,610 样本**（打牌 15,050；PASS 46,xxx；RIICHI 350；RON 63；TSUMO 22）。
   数据生成 49 s。
2. **加权 BC**（`train_teacher_bc.py`）：
   - 模型 `MahjongNet(channels=64, n_blocks=4, n_actions=47, touch_dim=8,
     extra_scalar=12)` = **453,232 参数**；
   - 样本权重按**决策类型**：打牌 **3–4×**、PASS **0.1→0.05×**、
     TSUMO/RON/RIICHI **20→30×**；
   - 理由：数据里 PASS 占 **75%**，不压制时模型只学 PASS（v1 实测打牌准确率
     49.5%、且永远推进不到听牌）。
3. **验收**：`eval_hanchan.py`（`play_model` **没有任何规则接管**，纯 argmax），
   10 半庄 = 80 局，与基线同种子。

## 3. 读数（同种子 80 局，CPU）

| 参赛 | 和牌局数 | 和牌率 | 备注 |
|---|---|---|---|
| uniform-random | 0/80 | 0.0% | 下界 |
| shanten-greedy（教师） | 33/80 | **41.25%** | 上界参考 |
| a2c_r11_it5（旧半庄 RL） | 2/80 | 2.5% | 改造前基线 |
| **teacher47_v1**（权重只给 special） | 2/40 | 5.0% | PASS 75% 淹没训练，打牌 acc 49.5% |
| **teacher47_v2**（加决策类型权重） | 12/80 | **15.0%** | 打牌 acc 78.7%，special recall 100% |
| **teacher47_v3**（v2 续训 + 更强权重） | 14/80 | **17.5%** | val_acc 95.2% |

**定性验证**（v3，40 局 11,277 决策点）：
- 可立直时选中立直 **25/25 = 100%**（不再靠规则接管）；
- 可和（自摸/荣和）时选中和牌 **3/3 = 100%**；
- 该批 40 局自身和 3 局（7.5%，种子不同，样本小）。

## 4. 训练曲线（v2/v3）

```
v2: epoch0 val_acc .843 -> epoch7 .945；打牌准确率 78.7%、RIICHI/RON/TSUMO recall 1.000
v3(续训, lr 5e-4): epoch0 .944 -> epoch5 .952；val 持续微升但和牌率 15%->17.5%（+2.5pt）
```

## 5. 结论与下一步

- **目标达成一半**：模型已能自己立直/和牌（选择率 100%），和牌率
  2.5% → 17.5%（7×），但离 >20% 目标差 2.5pt（80 局标准误约 ±4%，属边缘）。
- 剩余差距主要在**打牌质量**（打牌 imitation acc 78.7%，教师 41% vs 模型 17.5%）。
- 继续提升的优先级：
  1. **更多教师数据**：30→100 半庄（62k→200k 样本），打牌样本是瓶颈；
  2. **DAgger**：用当前模型跑局、教师在模型到过的状态打标（现在是纯 BC，
     分布偏移在 80 局里会累积）；
  3. **加大模型**（64×4→96×6）与训练轮数（当前 val_acc 还在缓慢上升）；
  4. 把「固定槽位」推广到 54 槽线（`train_reinforce.slot_of` 换成 47 槽同款），
     否则 54 槽线（riichi_net / from_games_*）永远受哈希冲突拖累。

## 6. 复现

```bash
cd /mnt/e/riichi-rl
PY=/mnt/d/WorkPlace/my_rl_project/venv/bin/python

$PY gen_teacher_data.py --hanchans 30 --out data/teacher47.npz      # 49 s
$PY train_teacher_bc.py --data data/teacher47.npz --epochs 8 \
    --special-weight 20 --discard-weight 3 --pass-weight 0.1 \
    --out checkpoints/teacher47_v2.pt                               # 约 13 min CPU
$PY train_teacher_bc.py --data data/teacher47.npz \
    --init checkpoints/teacher47_v2.pt --epochs 6 --lr 5e-4 \
    --special-weight 30 --discard-weight 4 --pass-weight 0.05 \
    --out checkpoints/teacher47_v3.pt                               # 约 12 min CPU
$PY eval_hanchan.py --ckpt checkpoints/teacher47_v3.pt --hanchans 10  # 模型 14/80
```

注：`data/teacher47.npz`（4.7 MB）与 `checkpoints/teacher47_*.pt`（1.8 MB/个）
均可再生，已加入 `.gitignore`，不入库。