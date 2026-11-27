# 会话笔记 2026-11-28: 修复包应用+验证

## 修复包来源与可信度
- `riichi-rl-fixed.zip`（192.168.137.1 Downloads，1.1M，md5 c6cbfff8b77b5d58d1b9f378a373a）
- 日期 2026-11-27 22:18，含 tests/test_engine_fixes_20261127.py（5个回归测试）
- 应用方式: rsync 覆盖（不删额外文件），已 commit e8440ce + push

## 5 大 bug 修复 + 验证状态
1. shanten 公式 -> 委托 shanten_ref(开源)
   验证: 随机2000手牌 vs refengine 0 差异(修复前95.9%错)
   关键: mahjong/shanten.py 是死代码(0引用), 真实走 refengine/fast_cache
2. 结算: 庄家倍数(12000对)/跳满倍满三倍满梯度/雀头符白发中/三暗刻2番二盃口3番/里宝牌时机/本场棒
   验证: 40符5番庄家 12000(修复前24000)
3. 食替: 碰7s后当巡禁切7s, PASS
4. _rule_override: 立直不强制接管(交模型选默听), 和牌仍接管
   验证: 3场景全对
5. reselect: 保证含后一半随机替换, PASS

## 测试结果
- test_engine_fixes_20261127: 5/5 PASS
- 全套: 152 passed, 2 failed(dual_stream/47_slot_ckpt, 均为修复前既有, git stash 验证过)

## 清理完成
- 24 日志 + 8 pycache + 206 中间iter checkpoint(390M) + 临时备份目录
- 保留 84 个最终模型, 工作区干净 HEAD=e8440ce

## 重要纠正(报告的因果链错误)
- shanten bug 不影响历史实验(死代码)
- 副露0%: greedy_tenpai 教师主动PASS(88-90行) + KL锚定 + PASS零成本 三重锁
  不是单纯"训练数据问题"; 且副露负收益(-4.8pt)是旧bug代码下测的, 修复后需重测

## 修复后需要重测的项
- ELO 全表(arena_fixed_elo 跑中, 旧值: dest_prob_v3 1740/eff_wuke_d2 1476)
- 副露 A/B(修复后副露价值可能变化)
- hand_predictor win_prob(依赖 shanten 特征, 需重采数据)

## 待办
- arena_fixed_elo 后台跑中(~15min), 完成后对比旧基线
