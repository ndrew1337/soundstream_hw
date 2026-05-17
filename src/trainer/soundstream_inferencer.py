import torch
import torchaudio
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.inferencer import Inferencer
from src.utils.audio_viz import compute_pair_visuals, save_pair_artifacts


class SoundStreamInferencer(Inferencer):
    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        self.sample_rate = config.inferencer.get("sample_rate", 16000)
        self.save_audio = config.inferencer.get("save_audio", True)
        self.audio_mean = config.inferencer.get("audio_mean", 0.0)
        self.audio_std = config.inferencer.get("audio_std", 1.0)
        self.save_visuals_count = int(config.inferencer.get("save_visuals_count", 0))
        self.visuals_n_fft = int(config.inferencer.get("visuals_n_fft", 1024))
        self.visuals_hop = int(config.inferencer.get("visuals_hop", 128))
        self.visuals_n_mels = int(config.inferencer.get("visuals_n_mels", 80))
        self.visuals_f_min = float(config.inferencer.get("visuals_f_min", 80.0))
        self.visuals_f_max = config.inferencer.get("visuals_f_max")
        self.visuals_mel_dpi = int(config.inferencer.get("visuals_mel_dpi", 140))
        self.visuals_mel_figure_w = float(config.inferencer.get("visuals_mel_figure_w", 12.0))
        self.visuals_mel_figure_h = float(config.inferencer.get("visuals_mel_figure_h", 4.5))
        self.visuals_mel_interpolation = config.inferencer.get(
            "visuals_mel_interpolation", "bilinear"
        )
        self._visuals_saved = 0
        super().__init__(
            model=model,
            config=config,
            device=device,
            dataloaders=dataloaders,
            save_path=save_path,
            metrics=metrics,
            batch_transforms=batch_transforms,
            skip_model_load=skip_model_load,
        )

    def _from_pretrained(self, pretrained_path):
        pretrained_path = str(pretrained_path)
        print(f"Loading model weights from: {pretrained_path} ...")
        checkpoint = torch.load(pretrained_path, map_location=self.device, weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)
        self.model.load_state_dict(state)

    def process_batch(self, batch_idx, batch, metrics, part):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        x_real = batch["audio"]
        x_fake, vq_loss, indices = self.model(x_real)
        batch["x_fake"] = x_fake
        batch["vq_loss"] = vq_loss
        batch["indices"] = indices

        if metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        if self.save_path is not None:
            batch_size = x_real.shape[0]
            current_id = batch_idx * batch_size
            for i in range(batch_size):
                length = int(batch["audio_len"][i].item()) if "audio_len" in batch else x_real.shape[-1]
                wave_fake = x_fake[i, :, :length].detach().cpu()
                wave_fake = wave_fake * self.audio_std + self.audio_mean
                wave_real = x_real[i, :, :length].detach().cpu()
                wave_real = wave_real * self.audio_std + self.audio_mean

                if self.save_audio:
                    out_path = self.save_path / part / f"recon_{current_id + i:06d}.wav"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    torchaudio.save(str(out_path), wave_fake, self.sample_rate)

                if self._visuals_saved < self.save_visuals_count:
                    visuals = compute_pair_visuals(
                        x_real=wave_real,
                        x_fake=wave_fake,
                        sample_rate=self.sample_rate,
                        n_fft=self.visuals_n_fft,
                        hop_length=self.visuals_hop,
                        n_mels=self.visuals_n_mels,
                        mel_f_min=self.visuals_f_min,
                        mel_f_max=self.visuals_f_max,
                        mel_dpi=self.visuals_mel_dpi,
                        mel_fig_width=self.visuals_mel_figure_w,
                        mel_fig_height=self.visuals_mel_figure_h,
                        mel_interpolation=self.visuals_mel_interpolation,
                    )
                    save_pair_artifacts(
                        out_dir=self.save_path / part / "visuals",
                        sample_id=current_id + i,
                        visuals=visuals,
                        sample_rate=self.sample_rate,
                    )
                    self._visuals_saved += 1

        return batch

    def _inference_part(self, part, dataloader):
        self.is_train = False
        self.model.eval()
        if self.evaluation_metrics is not None:
            self.evaluation_metrics.reset()

        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

        return self.evaluation_metrics.result() if self.evaluation_metrics is not None else {}
