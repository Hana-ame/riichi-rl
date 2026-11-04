"""统一测试入口：发现并运行 tests/ 下全部 test_*.py。

用法：
    python -m tests.run_all          # 全部
    python -m tests.test_env_core    # 单个模块（每个文件也可单独跑）

零依赖：只用标准库 unittest。
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    suite = unittest.TestLoader().discover(
        os.path.join(ROOT, "tests"), pattern="test_*.py", top_level_dir=ROOT)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print()
    print("=" * 64)
    print(f"通过 {result.testsRun - len(result.failures) - len(result.errors)}"
          f" / {result.testsRun}，失败 {len(result.failures)}，"
          f"错误 {len(result.errors)}")
    print("=" * 64)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())