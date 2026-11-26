"""麻将策略网络：观测编码 + 动作头。

架构参考（2026-10-04 调研）：
- Mortal 用 Conv1d 残差块 + 通道注意力，打牌头用排列组合打分
- LLM_Mahjong bc49 用 ConvFormer（d=160/6层/5头）+ 46 槽动作头，**2.00M 参数**

本实现按他们的结论选型：
- 观测重构成 (C, 34) 二维张量，卷积能捕捉「牌种之间的关系」
- 参数量控制在 2M 量级（bc49 证明麻将不需要更大）
- 动作头支持两种模式：
  - per_tile_head=False（默认，与旧 checkpoint 完全兼容）
  - per_tile_head=True（推荐新训练使用）：34 个切牌槽直接由逐张牌特征图
    (B, C, 34) 融合全局上下文输出，避免 AdaptiveAvgPool1d(4) 丢掉单张牌位置信息。

本文件只定义网络与编码，不含任何人类麻将知识。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mahjong import obs2 as OBS
from mahjong.tiles import NUM_TILES

#: 观测平面对数（每类 34 维，通道优先）
N_PLANES = OBS.N_PLANES
PLANE_SIZE = OBS.PLANE


class ChannelAttention(nn.Module):
    """通道注意力（Mortal 式，avg + max 双路）。"""

    def __init__(self, channels, ratio=4):
        super().__init__()
        hidden = max(4, channels // ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )

    def forward(self, x):            # x: (B, C, L)
        w = self.mlp(x.mean(-1) + x.amax(-1)).sigmoid()   # (B, C)
        return x * w.unsqueeze(-1)


class ResBlock(nn.Module):
    """1D 残差块（Mortal 式，pre-act）。"""

    def __init__(self, ch):
        super().__init__()
        self.norm1 = nn.GroupNorm(4, ch)
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(4, ch)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1, bias=False)
        self.attn = ChannelAttention(ch)

    def forward(self, x):
        h = F.relu(self.norm1(x))
        h = self.conv1(h)
        h = F.relu(self.norm2(h))
        h = self.conv2(h)
        return x + self.attn(h)


class MahjongNet(nn.Module):
    """策略网络。

    obs (B, OBS_DIM) -> 拆成 (B, N_PLANES, 34) + 标量 -> Conv trunk -> 动作 logits
    """

    def __init__(self, channels=160, n_blocks=6, n_actions=54,
                 touch_dim=0, extra_scalar=0, in_planes=None, aux_dim=0,
                 per_tile_head=False, scalar_dim=None):
        super().__init__()
        self.aux_dim = int(aux_dim or 0)
        self.aux_head = nn.Linear(256, self.aux_dim) if self.aux_dim else None
        self.in_planes = int(in_planes) if in_planes else N_PLANES
        self.n_planes = self.in_planes
        # scalar_dim 默认取当前 obs2（29 平面版）；legacy ckpt（in_planes=22）由
        # 调用方显式传入旧 obs2_v1 的 26，使 head_fc 输入维与旧权重对齐。
        self.scalar_dim = int(scalar_dim) if scalar_dim else OBS.SCALAR_DIM
        self.touch_dim = touch_dim
        self.in_scalar = self.scalar_dim + touch_dim + extra_scalar
        self.n_actions = int(n_actions)
        self.per_tile_head = bool(per_tile_head)

        self.embed = nn.Conv1d(self.n_planes, channels, kernel_size=1)
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(n_blocks)])

        # 池化：全局 + 分段（保留牌种序列的结构信息）
        self.head_conv = nn.Conv1d(channels, channels, 3, padding=1)
        self.segpool = nn.AdaptiveAvgPool1d(4)
        self.head_fc = nn.Sequential(
            nn.Linear(channels * 5 + self.in_scalar, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )
        if self.per_tile_head:
            # 逐牌切牌头：保留 (B, C, 34) 空间分辨率，再注入全局局势上下文
            self.tile_ctx = nn.Linear(256, channels)
            self.tile_policy = nn.Conv1d(channels, 1, kernel_size=1)
            self.special_policy = nn.Linear(256, self.n_actions - NUM_TILES)
            self.policy = None
        else:
            self.policy = nn.Linear(256, self.n_actions)
        self.value = nn.Linear(256, 1)

    def _encode_trunk(self, obs, touch=None):
        B = obs.shape[0]
        planes = obs[:, : self.n_planes * PLANE_SIZE].view(B, self.n_planes, PLANE_SIZE)
        scalars = obs[:, self.n_planes * PLANE_SIZE:]
        if touch is not None and self.touch_dim > 0:
            scalars = torch.cat([scalars, touch], dim=1)
        h = self.embed(planes)
        h = self.blocks(h)
        h = F.relu(self.head_conv(h))
        seg = self.segpool(h).flatten(1)
        glob = h.amax(-1)
        z = torch.cat([seg, glob, scalars], dim=1)
        return h, z

    def _trunk(self, obs, touch=None):
        _, z = self._encode_trunk(obs, touch)
        return z

    def _policy_logits(self, h, z_fc):
        if self.per_tile_head:
            ctx = self.tile_ctx(z_fc).unsqueeze(-1)
            discard_logits = self.tile_policy(F.relu(h + ctx)).squeeze(1)
            special_logits = self.special_policy(z_fc)
            return torch.cat([discard_logits, special_logits], dim=1)
        return self.policy(z_fc)

    def forward(self, obs, touch=None):
        """obs (B, DIM) -> (policy_logits, value)"""
        h, z = self._encode_trunk(obs, touch)
        z_fc = self.head_fc(z)
        return self._policy_logits(h, z_fc), self.value(z_fc).squeeze(-1)

    def forward_aux(self, obs, touch=None):
        """返回 (policy, value, aux_logits)，aux_dim=0 时 aux_logits=None。"""
        h, z = self._encode_trunk(obs, touch)
        z_fc = self.head_fc(z)
        aux = self.aux_head(z_fc) if self.aux_head is not None else None
        return self._policy_logits(h, z_fc), self.value(z_fc).squeeze(-1), aux


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = MahjongNet()
    x = torch.randn(2, OBS.OBS_TENSOR_DIM)
    logits, v = m(x)
    print(f"params={count_params(m):,}")
    print(f"obs={x.shape} -> logits={logits.shape} value={v.shape}")
