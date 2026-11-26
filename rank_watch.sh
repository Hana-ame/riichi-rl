#!/bin/bash
# 自动排位追踪：等 PPO 存出新 checkpoint，每个跑 Elo(300局) 记录排名趋势。
# 用法: setsid nohup bash rank_watch.sh > /tmp/rank_watch.log 2>&1 &
PY=/home/lumin/miniconda3/bin/python
CKDIR=/mnt/e/riichi-rl/checkpoints
PATTERN="ppo_rank_v2_it*.pt"
LAST=""
while true; do
  NEW=$(ls -t $CKDIR/$PATTERN 2>/dev/null | grep -E "it[0-9]+\.pt$" | head -1)
  if [ -n "$NEW" ] && [ "$NEW" != "$LAST" ]; then
    BASE=$(basename "$NEW" .pt)
    echo "=== [$(date +%H:%M:%S)] 新 ckpt: $BASE -> Elo 300局"
    cd /mnt/e/riichi-rl
    OMP_NUM_THREADS=2 timeout 3000 $PY -u arena_elo.py --games 600 --mode hanchan --tenhou --pin-extra \
      --extra-ckpt "$NEW" \
      --out /tmp/elo_${BASE}.md > /tmp/elo_${BASE}.log 2>&1
    # 抽取 ppo 自己 + 第1名的 Elo
    RANK=$(grep -oE "^\| [0-9]+ \| ppo_rank[^|]*\|[^|]*\| \*\*[0-9.]+\*\*" /tmp/elo_${BASE}.md | head -1)
    TOP1=$(grep -E "^\| 1 \|" /tmp/elo_${BASE}.md | head -1)
    echo "  ppo: $RANK"
    echo "  top1: $TOP1"
    LAST="$NEW"
  fi
  sleep 60
done
