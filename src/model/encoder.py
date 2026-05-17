import torch.nn as nn
import torch.nn.functional as F

from src.model.causal_conv import CausalConv1d
from src.model.residual_unit import ResidualUnit


class EncoderBlock(nn.Module):
    def __init__(self, N, S):
        super().__init__()
        self.elu = nn.ELU()
        self.res_unit1 = ResidualUnit(N // 2, dilation=1)
        self.res_unit2 = ResidualUnit(N // 2, dilation=3)
        self.res_unit3 = ResidualUnit(N // 2, dilation=9)
        self.conv = CausalConv1d(
            kernel_size=2 * S,
            in_channels=N // 2,
            out_channels=N,
            stride=S,
        )

    def forward(self, x):
        x = self.res_unit1(x)
        x = self.res_unit2(x)
        x = self.res_unit3(x)
        x = self.conv(self.elu(x))
        return x


class Encoder(nn.Module):
    def __init__(self, C, K):
        super().__init__()
        self.conv1 = CausalConv1d(kernel_size=7, in_channels=1, out_channels=C)
        self.encoder_blocks = nn.Sequential(
            EncoderBlock(2 * C, 2),
            EncoderBlock(4 * C, 4),
            EncoderBlock(8 * C, 5),
            EncoderBlock(16 * C, 5),
        )
        self.conv2 = CausalConv1d(kernel_size=3, in_channels=16 * C, out_channels=K)

    def forward(self, x):
        x = self.conv1(x)
        x = self.encoder_blocks(x)
        x = self.conv2(F.elu(x))
        return x