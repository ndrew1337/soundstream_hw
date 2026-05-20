import torch.nn as nn

from src.model.causal_conv import CausalConv1d


class ResidualUnit(nn.Module):
    def __init__(self, N, dilation, causal=True):
        super().__init__()
        self.elu = nn.ELU()
        self.conv1 = CausalConv1d(
            kernel_size=7, in_channels=N, out_channels=N, dilation=dilation, causal=causal
        )
        self.conv2 = CausalConv1d(
            kernel_size=1, in_channels=N, out_channels=N, causal=causal
        )

    def forward(self, x):
        return x + self.conv2(self.elu(self.conv1(self.elu(x))))
