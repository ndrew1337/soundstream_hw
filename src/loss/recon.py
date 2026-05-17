import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class MultiScaleMelLoss(nn.Module):
    def __init__(self, sample_rate=16000, scales=(6, 7, 8, 9, 10, 11), n_mels=64, eps=1e-5):
        super().__init__()
        self.scales = scales
        self.eps = eps
        self.mels = nn.ModuleList()
        for s in scales:
            win = 2 ** s
            hop = max(1, win // 4)
            mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=win,
                hop_length=hop,
                win_length=win,
                n_mels=n_mels,
                power=1.0,
            )
            self.mels.append(mel)

    def forward(self, x_real, x_fake):
        x_real = x_real.squeeze(1)
        x_fake = x_fake.squeeze(1)
        loss = 0.0
        for s, mel in zip(self.scales, self.mels):
            mel_real = mel(x_real)
            mel_fake = mel(x_fake)
            log_real = torch.log(mel_real + self.eps)
            log_fake = torch.log(mel_fake + self.eps)
            win = 2 ** s
            alpha = (win / 2.0) ** 0.5
            loss = loss + F.l1_loss(mel_fake, mel_real) + alpha * F.mse_loss(log_fake, log_real)
        return loss
