# 旧 ckpt 因 obs2 演进无法评测（2026-11-26）

## 现象
```
eval_selfplay_gate.py --ckpt teacher47_eff_wuke_d2 --mode hanchan --deals 2400
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x610 and 366x512)
```

## 根因
`version_benchmark.AutoAgent.__init__` 从 head_fc 输入维反推特征模式：
```python
head_in = sd["head_fc.0.weight"].shape[1]     # = 366
in_scalar = head_in - ch*5                     # = 366 - 320 = 46
extra_total = in_scalar - O2.SCALAR_DIM        # = 46 - 32 = 14  > 0
# → 误判成 hanchan 模式（touch8+sit12），encoder 输出 610 维，ckpt 期望 366
```

ckpt 元数据：
- channels=64, blocks=4, in_planes=22, obs_order=False
- head_fc.0.weight = (512, **366**) → 64ch×5(320) + **46 标量**

当前 `mahjong/obs2.py`：
- N_PLANES=29, SCALAR_DIM=**32**

即 `teacher47_eff_wuke_d2` 是用**旧 obs2（22 平面 + 46 标量）**训练的。该 SCALAR_DIM=46
版本**不在 git 历史**里（`git log -S "SCALAR_DIM = 46"` 无结果，只有 26→32 的迁移记录），
是本地 zip 合入时被覆盖的中间版本。

## 影响
- `teacher47_eff_wuke_d2`（此前 best，半庄 1200 局 15.81% CI[14.8,16.9]）**当前不可复现评测**
- 任何 in_planes=22 / 标量维≠32 的旧 ckpt 都同样失效
- 新 ckpt（in_planes=29、SCALAR_DIM=32）不受影响，可正常评测

## 可行出路（未实施）
1. **修 AutoAgent 反推**：优先信 `ck.get("in_planes")` 与已知 (N_PLANES, SCALAR_DIM) 组合表，
   而非用 `extra_total>0` 猜 hanchan；对 in_planes=22 且 head_in=ch*5+46 的组合显式映射回
   旧 obs2 标量构造。需重建旧 46 标量 obs2 builder（git 里没有，需从 ckpt 反推特征语义）。
2. **重训 best**：用当前 obs2（29 平面/32 标量）重跑 eff 教师线 + DAgger，恢复一个可评测的
   best 候选。成本高（教师 BC 多轮）。
3. **接受新 ckpt 基线**：以 tenhou_big128_v3 等新架构模型为当前可评测基线。

## 结论
和牌 CI 下界 ≥15 的冲线目标当前**受阻**：best 模型因 obs2 演进不可评测，新模型实测远低于
门槛（tenhou_bc 系列 2-3%）。在 obs2 旧 builder 恢复或 best 重训之前，无法继续收窄 CI。
