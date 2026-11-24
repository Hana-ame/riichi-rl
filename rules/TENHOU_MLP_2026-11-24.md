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
