import torch
from torchmetrics.audio.nisqa import NonIntrusiveSpeechQualityAssessment
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

from src.metrics.base_metric import BaseMetric


class STOIMetric(BaseMetric):
    def __init__(
        self,
        sample_rate=16000,
        device="auto",
        denormalize_mean=None,
        denormalize_std=None,
        log_every_n_epochs=1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = ShortTimeObjectiveIntelligibility(fs=sample_rate, extended=False).to(device)
        self.denormalize_mean = denormalize_mean
        self.denormalize_std = denormalize_std
        self.log_every_n_epochs = log_every_n_epochs

    def _denormalize(self, audio):
        if self.denormalize_mean is None or self.denormalize_std is None:
            return audio
        return audio * self.denormalize_std + self.denormalize_mean

    def __call__(self, audio, x_fake, audio_len=None, **kwargs):
        target = self._denormalize(audio).squeeze(1)
        preds = self._denormalize(x_fake).squeeze(1)
        if audio_len is not None:
            length = int(audio_len.min().item())
            target = target[..., :length]
            preds = preds[..., :length]
        return self.metric(preds, target)


class NISQAMetric(BaseMetric):
    def __init__(
        self,
        sample_rate=16000,
        device="auto",
        denormalize_mean=None,
        denormalize_std=None,
        log_every_n_epochs=1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = NonIntrusiveSpeechQualityAssessment(fs=sample_rate).to(device)
        self.denormalize_mean = denormalize_mean
        self.denormalize_std = denormalize_std
        self.log_every_n_epochs = log_every_n_epochs

    def _denormalize(self, audio):
        if self.denormalize_mean is None or self.denormalize_std is None:
            return audio
        return audio * self.denormalize_std + self.denormalize_mean

    def __call__(self, x_fake, audio_len=None, **kwargs):
        preds = self._denormalize(x_fake).squeeze(1)
        if audio_len is not None:
            length = int(audio_len.min().item())
            preds = preds[..., :length]
        scores = self.metric(preds)
        return scores[0]
