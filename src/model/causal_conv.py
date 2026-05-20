import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """1D conv with causal or symmetric ("same") padding.

    Total padding stays the same in both modes, so output lengths match.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        stride=1,
        padding=0,
        causal=True,
    ):
        super().__init__()
        self.causal = causal
        total_pad = (kernel_size - 1) * dilation
        if causal:
            self.pad = (total_pad, 0)
        else:
            left = total_pad // 2
            self.pad = (left, total_pad - left)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        return self.conv(F.pad(x, self.pad))


class CausalConvTranspose1d(nn.Module):
    """Transposed 1D conv with causal or symmetric trimming."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, causal=True):
        super().__init__()
        self.causal = causal
        self.stride = stride
        self.trim_total = kernel_size - stride
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )

    def forward(self, x):
        x = self.conv_transpose(x)
        if self.trim_total > 0:
            if self.causal:
                x = x[..., : -self.trim_total]
            else:
                trim_left = self.trim_total // 2
                trim_right = self.trim_total - trim_left
                x = x[..., trim_left : x.shape[-1] - trim_right]
        return x
