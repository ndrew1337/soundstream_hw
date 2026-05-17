import torch.nn as nn

from src.loss.gan import (
    discriminator_hinge_loss,
    feature_matching_loss,
    generator_hinge_loss,
)
from src.loss.recon import MultiScaleMelLoss


class SoundStreamGeneratorLoss(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        lambda_recon=1.0,
        lambda_adv=1.0,
        lambda_fm=100.0,
        lambda_commit=1.0,
    ):
        super().__init__()
        self.mel_loss = MultiScaleMelLoss(sample_rate=sample_rate)
        self.lambda_recon = lambda_recon
        self.lambda_adv = lambda_adv
        self.lambda_fm = lambda_fm
        self.lambda_commit = lambda_commit

    def forward(self, x_real, x_fake, real_outputs, fake_outputs, vq_loss, **kwargs):
        rec = self.mel_loss(x_real, x_fake)
        adv = generator_hinge_loss(fake_outputs)
        fm = feature_matching_loss(real_outputs, fake_outputs)
        loss = (
            self.lambda_recon * rec
            + self.lambda_adv * adv
            + self.lambda_fm * fm
            + self.lambda_commit * vq_loss
        )
        return {
            "loss": loss,
            "loss_recon": rec,
            "loss_adv": adv,
            "loss_fm": fm,
            "loss_commit": vq_loss,
        }


class SoundStreamDiscriminatorLoss(nn.Module):
    def forward(self, real_outputs, fake_outputs, **kwargs):
        loss = discriminator_hinge_loss(real_outputs, fake_outputs)
        return {"loss_d": loss}
