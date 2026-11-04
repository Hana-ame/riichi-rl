"""全局硬约束：**只用 CPU，禁止 GPU**。

用户 2026-10-05 指示：「使用 CPU 而不是 GPU，禁止 GPU 使用」。

实现要点（不是 `torch.cuda.is_available()` 就完事）：
1. `CUDA_VISIBLE_DEVICES=""` 必须在 **import torch 之前**设好，
   否则 torch 初始化时已经绑定了设备；
2. 之后仍然显式 `.to("cpu")`，双保险——万一环境变量被外部覆盖，
   代码本身也不会走到 GPU。

必须在本模块里 `import torch` **之前**设环境变量，所以这个文件
要在任何 `import torch` 的模块之前 import。
"""

import os

# 必须早于 torch 初始化
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import torch  # noqa: E402

DEVICE = "cpu"

torch.set_num_threads(max(1, (os.cpu_count() or 4)))


def set_inference_threads(n=2):
    """在线推理（逐条决策）把线程压到 1–2。

    `rules/MODEL_BENCH.md` §4 实测：本机 batch=1 下 12 线程反而比 1 线程
    慢 4–7×（OpenMP 自旋 + 长尾到 0.5 s）；批量前向才是 4–8 线程更快。
    观战/竞技场都是逐决策推理，必须调这个。
    """
    torch.set_num_threads(max(1, int(n)))


def set_batch_threads(n=6):
    """批量训练/自对弈前向用 4–8 线程（见 MODEL_BENCH §4.2）。"""
    torch.set_num_threads(max(1, int(n)))


def device():
    """返回唯一的合法设备 'cpu'。"""
    return DEVICE


def assert_cpu_only():
    """训练/评测前的硬校验：CUDA 必须不可用。"""
    if torch.cuda.is_available():
        raise RuntimeError(
            "检测到 CUDA 可用，但本项目硬约束为禁止 GPU 使用。"
            "（CUDA_VISIBLE_DEVICES 应为空）")
    return DEVICE


def load_for_cpu(obj):
    """任何 .to()/map_location 都强制落到 CPU。"""
    if isinstance(obj, torch.nn.Module):
        return obj.to("cpu")
    if isinstance(obj, dict):
        return {k: load_for_cpu(v) for k, v in obj.items()}
    if torch.is_tensor(obj):
        return obj.to("cpu")
    return obj