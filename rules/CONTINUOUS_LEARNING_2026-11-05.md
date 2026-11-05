# 持续学习 / 数据采集 / 数据增量添加闭环（2026-11-05）

> 回应审阅「未见持续学习闭环、replay/数据集采集、数据增量添加的独立证据」。

## 闭环总览

```
对局/观战采集 (serve_game.rec_step/flush_rec → data/selfplay/*.npz)
        │  （703 局、自动续号，含 obs/act/hands/rivers/melds/riichi/结果）
        ▼
教师/DAgger 标注 (gen_teacher_data.py / dagger_collect.py → *_mix.npz)
        │  教师可为 greedy / call_first / hybridd / Suphx 三种头
        ▼
训练 (train_teacher_bc.py BC/DAgger; train_hanchan.py REINFORCE/PPO;
        train_aux_heads.py 辅助头)
        ▼
评测 (eval_hanchan.py 80/40 局；tenpai_audit.py 60 局/席；head2head.py;
        realtime_bench.py)
        ▼
登记与闸门 (version_registry.py 91 ckpt；version_ladder.py 同池 Elo + 95% CI)
        │
        └──> 新的强/弱结论写回 rules/*，决定下一轮采什么数据（增量）
```

## 增量数据的具体落点（已有文件/commit）

| 环节 | 文件 | 证据/读数 |
|---|---|---|
| 对局采集 | `serve_game.py` `rec_step/flush_rec` | selfplay 703 局、11M、自动续号 |
| 人类牌谱 | `data/houou` 528M → `build_cache.py` → `data/cache` 585M → `bc_v1`（243 槽） | bc_v1 训练/评测在 ELO 600 局中 1666.3 |
| 教师数据 | `gen_teacher_data.py --hanchans 100/300` | 100 半庄 208,542 样本；300 半庄 624,001 样本 |
| DAgger | `dagger_collect.py --teacher {greedy,call_first,hybridd,bc,suphx,suphxfull}` | 120/150/300/500/600 局多轮；样本格式与教师数据一致 |
| BC 训练 | `train_teacher_bc.py --data/--init/--epochs/--aux-weights` | 100h_v2 val_acc 0.962；300h val 0.976（但实战降） |
| RL 训练 | `train_hanchan.py`（REINFORCE/A2C 风格 + `--ppo-epochs/--clip` PPO） | 3 轮 RL 全负（见 RL_PILOT/PPO 文档） |
| 辅助头 | `aux_heads.py --train` | 65,696 样本、607 局；MAE 0.671/AUC 0.996/win 0.572 |
| 评测 | `eval_hanchan.py`、`tenpai_audit.py`、`head2head.py`、`realtime_bench.py` | 每局每座位口径 + 5s 实时性 |
| 登记/闸门 | `version_registry.py`（91 ckpt/11 血缘）、`version_ladder.py`（500 局、CI 闸门） | 100h_v2 天梯 FAIL 不发布 |

## 增量添加的实操（本轮已跑过多次）

- 新数据：`data/dagger_*`（多个版本，120–600 局）→ 合并 `*_mix.npz` → `train_teacher_bc --init 100h_v2`；
- 新版本：每轮产出 `teacher47_*` ckpt → `version_registry` 自动纳入 → `version_ladder` 同池比较；
- 结论回流：CEL/DAgger 四教师 + 300 半庄 BC + 3 轮 RL 的负结果全部写入 `rules/*`，
  下一轮只采「外部强模型教师」数据（这正是当前 600 局蒸馏在做的事）。

## 当前持续学习状态

- 闭环**在跑**：本轮启动的 `suphxfull` 教师 600 局采集 → 纯 DAgger 训练 2 epochs →
  80 局 + 同池审计（后台），就是把「外部强模型」增量数据加进主策略的闭环实例；
- 闭环**未证明有效**：所有增量尝试目前都未把座位和牌率推过 15%（结果已全部落档）。
