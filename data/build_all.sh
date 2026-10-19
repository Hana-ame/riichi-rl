#!/bin/bash
# 并行编译全部 houou shard -> data/cache/<year>_<chunk>.npz
#
# 用 data/build_one.sh 做包装层，理由见该文件注释：
# 不能把 $(basename ...) 直接写在 xargs -I{} 的命令行里。
cd /mnt/e/riichi-rl || exit 1
mkdir -p data/cache
ls /mnt/e/riichi-rl/data/houou/*.tar \
  | xargs -P 10 -I{} bash /mnt/e/riichi-rl/data/build_one.sh "{}"
echo "CACHE_DONE n=$(ls /mnt/e/riichi-rl/data/cache/*.npz 2>/dev/null | wc -l)"