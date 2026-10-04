"""麻将策略网络：观测编码 + 动作头。

架构参考（2026-10-04 调研）：
- Mortal 用 Conv1d 残差块 + 通道注意力，打牌头用排列组合打分
- LLM_Mahjong bc49 用 ConvFormer（d=160/6层/5头）+ 46 槽动作头，**2.00M 参数**

本实现按他们的结论选型：
- 观测重构成 (C, 34) 二维张量，卷积能捕捉「牌种之间的关系」
- 参数量控制在 2M 量级（bc49 证明麻将不需要更大）
- 动作头用 slot 分类 + mask，不用逐张 softmax

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
                 touch_dim=0, extra_scalar=0):
        super().__init__()
        self.n_planes = N_PLANES
        self.scalar_dim = OBS.SCALAR_DIM
        self.touch_dim = touch_dim
        self.in_scalar = self.scalar_dim + touch_dim + extra_scalar

        self.embed = nn.Conv1d(N_PLANES, channels, kernel_size=1)
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
        self.policy = nn.Linear(256, n_actions)
        self.value = nn.Linear(256, 1)

    def forward(self, obs, touch=None):
        """obs (B, DIM) -> (policy_logits, value)

        touch: 可选的「上一手形状摘要」附加到标量段。

        动机：RL 若只从「输赢」学，需要几百万局才摸到手牌结构。
        把**自己手牌的形状统计**（听牌数、向听、面子/搭子计数）作为
        标量特征喂进网络，是把「局面是什么」显式化，不是牌效表——
        这些量对每个手牌都能算，不含任何「该打哪张」的知识。

        形状：touch_dim 维 float，与 obs 拼接后再进 head_fc。
        """
        B = obs.shape[0]
        planes = obs[:, : self.n_planes * PLANE_SIZE].view(B, self.n_planes, PLANE_SIZE)
        scalars = obs[:, self.n_planes * PLANE_SIZE:]

        if touch is not None and self.touch_dim > 0:
            scalars = torch.cat([scalars, touch], dim=1)

        h = self.embed(planes)
        h = self.blocks(h)
        h = F.relu(self.head_conv(h))

        # 四分段池化 + 全局池化（34 不能被 4 整除，用自适应池化）
        seg = self.segpool(h).flatten(1)
        glob = h.amax(-1)                                        # (B, C/2)
        z = torch.cat([seg, glob, scalars], dim=1)
        z = self.head_fc(z)

        return self.policy(z), self.value(z).squeeze(-1)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = MahjongNet()
    x = torch.randn(2, OBS.OBS_TENSOR_DIM)
    logits, v = m(x)
    print(f"params={count_params(m):,}")
    print(f"obs={x.shape} -> logits={logits.shape} value={v.shape}")
