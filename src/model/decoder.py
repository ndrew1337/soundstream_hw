import torch.nn as nn
import torch.nn.functional as F

from src.model.causal_conv import CausalConv1d, CausalConvTranspose1d
from src.model.residual_unit import ResidualUnit


class DecoderBlock(nn.Module):
    def __init__(self, N, S, causal=True):
        super().__init__()
        self.conv_transpose = CausalConvTranspose1d(
            in_channels=N,
            out_channels=N // 2,
            kernel_size=2 * S,
            stride=S,
            causal=causal,
        )
        self.res_unit1 = ResidualUnit(N // 2, dilation=1, causal=causal)
        self.res_unit2 = ResidualUnit(N // 2, dilation=3, causal=causal)
        self.res_unit3 = ResidualUnit(N // 2, dilation=9, causal=causal)

    def forward(self, x):
        x = self.conv_transpose(F.elu(x))
        x = self.res_unit1(x)
        x = self.res_unit2(x)
        x = self.res_unit3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, C, K, causal=True):
        super().__init__()
        self.conv1 = CausalConv1d(
            kernel_size=7, in_channels=K, out_channels=16 * C, causal=causal
        )
        self.decoder_blocks = nn.Sequential(
            DecoderBlock(16 * C, 5, causal=causal),
            DecoderBlock(8 * C, 5, causal=causal),
            DecoderBlock(4 * C, 4, causal=causal),
            DecoderBlock(2 * C, 2, causal=causal),
        )
        self.conv2 = CausalConv1d(kernel_size=7, in_channels=C, out_channels=1, causal=causal)

    def forward(self, x):
        x = self.conv1(x)
        x = self.decoder_blocks(x)
        x = self.conv2(F.elu(x))
        return x
