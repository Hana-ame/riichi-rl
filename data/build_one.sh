#!/bin/bash
# 单个 shard 编译包装：参数是 shard 的完整路径。
#
# 为什么要单独一层：之前把 "$(basename {} | sed ...)" 直接写在
# `xargs -I{}` 的命令行里，命令替换会先被**外层 shell**求值
# （那时 {} 还没被替换），于是输出路径变成
#   /mnt/e/riichi-rl/data/cache/mnt/e/riichi-rl/data/houou/2024_00007.tar.npz
# 全部 24 个 worker 写同一个路径 -> FileNotFoundError 且 0 产物。
src="$1"
name=$(basename "$src" .tar)
exec /home/lumin/miniconda3/bin/python /mnt/e/riichi-rl/build_cache.py \
  "$src" "/mnt/e/riichi-rl/data/cache/${name}.npz"