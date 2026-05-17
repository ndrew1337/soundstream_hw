import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class SubDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(1, 16, 15, 1, padding=7)),
            weight_norm(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            weight_norm(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            weight_norm(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            weight_norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            weight_norm(nn.Conv1d(1024, 1, 3, 1, padding=1)),
        ])
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        features = []
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i < len(self.convs) - 1:
                x = self.lrelu(x)
            features.append(x)
        return x, features


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            SubDiscriminator() for _ in range(3)
        ])
        self.pooling = nn.AvgPool1d(4, 2, padding=2)

    def forward(self, x):
        results = []
        for i, d in enumerate(self.discriminators):
            if i > 0:
                x = self.pooling(x)
            logits, feats = d(x)
            results.append((logits, feats))
        return results


class ResidualUnit2D(nn.Module):
    def __init__(self, in_channels, m, s):
        super().__init__()
        N = in_channels
        s_t, s_f = s
        self.conv1 = weight_norm(nn.Conv2d(N, N, kernel_size=3, padding=1))
        self.conv2 = weight_norm(nn.Conv2d(
            N, m * N,
            kernel_size=(s_t + 2, s_f + 2),
            stride=(s_t, s_f),
            padding=(1, 1),
        ))
        self.skip = weight_norm(nn.Conv2d(N, m * N, kernel_size=1, stride=(s_t, s_f)))
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = self.skip(x)
        x = self.lrelu(self.conv1(x))
        x = self.conv2(x)
        min_h = min(residual.shape[-2], x.shape[-2])
        min_w = min(residual.shape[-1], x.shape[-1])
        residual = residual[..., :min_h, :min_w]
        x = x[..., :min_h, :min_w]
        return self.lrelu(x + residual)


class STFTDiscriminator(nn.Module):
    def __init__(self, C=32, n_fft=1024, hop_length=256, win_length=1024):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

        self.first_conv = weight_norm(nn.Conv2d(2, C, kernel_size=7, padding=3))

        self.blocks = nn.ModuleList([
            ResidualUnit2D(C, m=2, s=(1, 2)),
            ResidualUnit2D(2*C, m=2, s=(2, 2)),
            ResidualUnit2D(4*C, m=1, s=(1, 2)),
            ResidualUnit2D(4*C, m=2, s=(2, 2)),
            ResidualUnit2D(8*C, m=1, s=(1, 2)),
            ResidualUnit2D(8*C, m=2, s=(2, 2)),
        ])
        self.final_conv = weight_norm(nn.Conv2d(16*C, 1, kernel_size=(1, 8)))
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = x.squeeze(1)
        stft = torch.stft(x, self.n_fft, self.hop_length, self.win_length, window=self.window, return_complex=True)
        stft = stft[:, 1:, :]
        ri = torch.stack([stft.real, stft.imag], dim=1)
        ri = ri.transpose(2, 3)

        features = []
        x = self.first_conv(ri)
        features.append(x)
        for block in self.blocks:
            x = block(x)
            features.append(x)
        logits = self.final_conv(x)
        return logits, features


class SoundStreamDiscriminator(nn.Module):
    def __init__(self, stft_n_fft=1024, stft_hop=256, stft_win=1024, stft_C=32):
        super().__init__()
        self.wave = MultiScaleDiscriminator()
        self.stft = STFTDiscriminator(C=stft_C, n_fft=stft_n_fft, hop_length=stft_hop, win_length=stft_win)

    def forward(self, x):
        outputs = self.wave(x)
        outputs = outputs + [self.stft(x)]
        return outputs