#!/bin/bash
# 等 PPO 训练完成后，跑 it100 和最新 ckpt 的 gate 每手口径
PY=/home/lumin/miniconda3/bin/python
while ps -p 321235 > /dev/null 2>&1; do sleep 30; done
echo "=== PPO 训练完成，开始 gate 评测 ==="
for CKPT in ppo_rank_v2_it100 ppo_rank_v2_it200; do
  if [ -f checkpoints/$CKPT.pt ]; then
    echo "=== gate $CKPT (400局每手口径) ==="
    OMP_NUM_THREADS=4 $PY -u eval_selfplay_gate.py --ckpt $CKPT --mode hanchan --deals 400 --seed0 500000 2>&1 | grep -E "口径|每手|门槛|场半庄"
  fi
done
echo "=== 全部完成 ==="
