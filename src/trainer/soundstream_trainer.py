import torch
from collections import OrderedDict
from torch.nn.utils import clip_grad_norm_

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer

from src.utils.audio_viz import compute_pair_visuals
from src.utils.io_utils import ROOT_PATH

class SoundStreamTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        discriminator,
        g_criterion,
        d_criterion,
        metrics,
        g_optimizer,
        d_optimizer,
        g_lr_scheduler,
        d_lr_scheduler,
        config,
        device,
        dataloaders,
        logger,
        writer,
        epoch_len=None,
        skip_oom=True,
        batch_transforms=None,
    ):
        super().__init__(
            model=model,
            criterion=g_criterion,
            metrics=metrics,
            optimizer=g_optimizer,
            lr_scheduler=g_lr_scheduler,
            config=config,
            device=device,
            dataloaders=dataloaders,
            logger=logger,
            writer=writer,
            epoch_len=epoch_len,
            skip_oom=skip_oom,
            batch_transforms=batch_transforms,
        )
        self.discriminator = discriminator.to(device)
        self.g_criterion = g_criterion.to(device)
        self.d_criterion = d_criterion.to(device)
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.g_lr_scheduler = g_lr_scheduler
        self.d_lr_scheduler = d_lr_scheduler
        self.log_audio_count = config.trainer.get("log_audio_count", 2)
        self.sample_rate = config.trainer.get("sample_rate", 16000)
        self.log_visuals_n_fft = config.trainer.get("log_visuals_n_fft", 1024)
        self.log_visuals_hop = config.trainer.get("log_visuals_hop", 128)
        self.log_visuals_n_mels = config.trainer.get("log_visuals_n_mels", 80)
        self.log_visuals_f_min = float(config.trainer.get("log_visuals_f_min", 80.0))
        self.log_visuals_f_max = config.trainer.get("log_visuals_f_max")
        self.log_visuals_dpi = int(config.trainer.get("log_visuals_dpi", 140))
        self.log_visuals_figure_w = float(config.trainer.get("log_visuals_figure_w", 12.0))
        self.log_visuals_figure_h = float(config.trainer.get("log_visuals_figure_h", 4.5))
        self.log_visuals_mel_interpolation = config.trainer.get(
            "log_visuals_mel_interpolation", "bilinear"
        )
        self.log_audio_peak_normalize_preview = bool(
            config.trainer.get("log_audio_peak_normalize_preview", True)
        )
        self.log_real_audio_once = bool(config.trainer.get("log_real_audio_once", True))
        self._logged_real_audio = {"train": False, "test": False}

        self.use_amp = bool(config.trainer.get("use_amp", False)) and "cuda" in str(device)
        self.amp_dtype = torch.float16
        self.g_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.d_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def process_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        x_real = batch["audio"]

        autocast = torch.autocast(
            device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp
        )

        if self.is_train:
            self.model.train()
            self.discriminator.train()

            with torch.no_grad(), autocast:
                x_fake_for_d, _, _ = self.model(x_real)

            with autocast:
                real_outputs = self.discriminator(x_real)
                fake_outputs = self.discriminator(x_fake_for_d.detach())
                d_losses = self.d_criterion(
                    real_outputs=real_outputs, fake_outputs=fake_outputs
                )

            self.d_optimizer.zero_grad()
            self.d_scaler.scale(d_losses["loss_d"]).backward()
            self.d_scaler.unscale_(self.d_optimizer)
            d_grad_norm = self._grad_norm(self.discriminator)
            self.d_scaler.step(self.d_optimizer)
            self.d_scaler.update()
            if self.d_lr_scheduler is not None:
                self.d_lr_scheduler.step()

            with autocast:
                x_fake, vq_loss, indices = self.model(x_real)
                real_outputs = self.discriminator(x_real)
                fake_outputs = self.discriminator(x_fake)
                g_losses = self.g_criterion(
                    x_real=x_real,
                    x_fake=x_fake,
                    real_outputs=real_outputs,
                    fake_outputs=fake_outputs,
                    vq_loss=vq_loss,
                )

            self.g_optimizer.zero_grad()
            self.g_scaler.scale(g_losses["loss"]).backward()
            self.g_scaler.unscale_(self.g_optimizer)
            self._clip_grad_norm()
            g_grad_norm = self._grad_norm(self.model)
            self.g_scaler.step(self.g_optimizer)
            self.g_scaler.update()
            if self.g_lr_scheduler is not None:
                self.g_lr_scheduler.step()

            batch.update(g_losses)
            batch.update(d_losses)
            batch["x_fake"] = x_fake.float()
            batch["indices"] = indices
            batch["g_grad_norm"] = g_grad_norm
            batch["d_grad_norm"] = d_grad_norm
        else:
            self.model.eval()
            self.discriminator.eval()
            with torch.no_grad(), autocast:
                x_fake, vq_loss, indices = self.model(x_real)
                real_outputs = self.discriminator(x_real)
                fake_outputs = self.discriminator(x_fake)
                g_losses = self.g_criterion(
                    x_real=x_real,
                    x_fake=x_fake,
                    real_outputs=real_outputs,
                    fake_outputs=fake_outputs,
                    vq_loss=vq_loss,
                )
                d_losses = self.d_criterion(
                    real_outputs=real_outputs, fake_outputs=fake_outputs
                )
            batch.update(g_losses)
            batch.update(d_losses)
            batch["x_fake"] = x_fake.float()
            batch["indices"] = indices

        for loss_name in self.config.writer.loss_names:
            if loss_name in batch and torch.is_tensor(batch[loss_name]):
                metrics.update(loss_name, batch[loss_name].item())

        if self.is_train:
            metrics.update("g_grad_norm", batch["g_grad_norm"])
            metrics.update("d_grad_norm", batch["d_grad_norm"])

        metric_funcs = self.metrics["train"] if self.is_train else self.metrics["inference"]
        current_epoch = max(int(getattr(self, "_last_epoch", 1)), 1)
        for met in metric_funcs:
            every = int(getattr(met, "log_every_n_epochs", 1))
            if every > 1 and current_epoch % every != 0:
                continue
            metrics.update(met.name, met(**batch))

        return batch

    def _clip_grad_norm(self):
        max_grad = self.config.trainer.get("max_grad_norm")
        if max_grad is not None:
            clip_grad_norm_(self.model.parameters(), max_grad)

    @torch.no_grad()
    def _grad_norm(self, module, norm_type=2.0):
        params = [p for p in module.parameters() if p.grad is not None]
        if not params:
            return 0.0
        norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in params]),
            norm_type,
        )
        return norm.item()

    def _log_batch(self, batch_idx, batch, mode="train"):
        if self.writer is None:
            return

        if "indices" in batch:
            indices = batch["indices"]
            num_codes = self.config.model.num_embeddings
            counts = torch.bincount(indices.flatten(), minlength=num_codes).float()
            probs = counts / counts.sum().clamp(min=1.0)
            perplexity = torch.exp(-(probs * (probs + 1e-10).log()).sum())
            self.writer.add_scalar("perplexity", perplexity.item())

        if "x_fake" in batch and "audio" in batch:
            x_real = batch["audio"]
            x_fake = batch["x_fake"]
            n = min(self.log_audio_count, x_real.shape[0])
            log_real = not (
                self.log_real_audio_once and self._logged_real_audio.get(mode, False)
            )
            group_fn = getattr(self.writer, "log_media_ordered", None)
            build_audio = getattr(self.writer, "build_audio", None)
            build_image = getattr(self.writer, "build_image", None)
            use_group = group_fn is not None and build_audio is not None and build_image is not None
            pn = self.log_audio_peak_normalize_preview

            for i in range(n):
                visuals = compute_pair_visuals(
                    x_real=x_real[i],
                    x_fake=x_fake[i],
                    sample_rate=self.sample_rate,
                    n_fft=self.log_visuals_n_fft,
                    hop_length=self.log_visuals_hop,
                    n_mels=self.log_visuals_n_mels,
                    mel_f_min=self.log_visuals_f_min,
                    mel_f_max=self.log_visuals_f_max,
                    mel_dpi=self.log_visuals_dpi,
                    mel_fig_width=self.log_visuals_figure_w,
                    mel_fig_height=self.log_visuals_figure_h,
                    mel_interpolation=self.log_visuals_mel_interpolation,
                )
                base = f"sample_{i:02d}"
                if use_group:
                    row = OrderedDict()
                    if log_real:
                        row[f"{base}_wave_input"] = build_audio(
                            visuals["wave_real"],
                            self.sample_rate,
                            caption="input",
                            peak_normalize_preview=False,
                        )
                    row[f"{base}_wave_reconstruction"] = build_audio(
                        visuals["wave_fake"],
                        self.sample_rate,
                        caption="reconstruction",
                        peak_normalize_preview=pn,
                    )
                    row[f"{base}_wave_difference"] = build_audio(
                        visuals["wave_diff"],
                        self.sample_rate,
                        caption="difference (reconstruction − input)",
                        peak_normalize_preview=pn,
                    )
                    if log_real:
                        row[f"{base}_mel_input"] = build_image(visuals["mel_real_image"])
                    row[f"{base}_mel_reconstruction"] = build_image(
                        visuals["mel_fake_image"]
                    )
                    row[f"{base}_mel_difference"] = build_image(visuals["mel_diff_image"])
                    group_fn(row)
                else:
                    if log_real:
                        self.writer.add_audio(
                            f"{base}_wave_input",
                            visuals["wave_real"],
                            self.sample_rate,
                            caption="input",
                        )
                    self.writer.add_audio(
                        f"{base}_wave_reconstruction",
                        visuals["wave_fake"],
                        self.sample_rate,
                        caption="reconstruction",
                        peak_normalize_preview=pn,
                    )
                    self.writer.add_audio(
                        f"{base}_wave_difference",
                        visuals["wave_diff"],
                        self.sample_rate,
                        caption="difference (reconstruction − input)",
                        peak_normalize_preview=pn,
                    )
                    if log_real:
                        self.writer.add_image(
                            f"{base}_mel_input", visuals["mel_real_image"]
                        )
                    self.writer.add_image(
                        f"{base}_mel_reconstruction", visuals["mel_fake_image"]
                    )
                    self.writer.add_image(
                        f"{base}_mel_difference", visuals["mel_diff_image"]
                    )
            if log_real:
                self._logged_real_audio[mode] = True

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "g_optimizer": self.g_optimizer.state_dict(),
            "d_optimizer": self.d_optimizer.state_dict(),
            "g_lr_scheduler": self.g_lr_scheduler.state_dict()
                if self.g_lr_scheduler is not None else None,
            "d_lr_scheduler": self.d_lr_scheduler.state_dict()
                if self.d_lr_scheduler is not None else None,
            "g_scaler": self.g_scaler.state_dict() if self.use_amp else None,
            "d_scaler": self.d_scaler.state_dict() if self.use_amp else None,
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        filename = str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth")
        if not (only_best and save_best):
            torch.save(state, filename)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(filename, str(self.checkpoint_dir.parent))
            self.logger.info(f"Saving checkpoint: {filename} ...")
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(best_path, str(self.checkpoint_dir.parent))
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path, map_location=self.device, weights_only=False)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]
        self.model.load_state_dict(checkpoint["state_dict"])
        if "discriminator_state_dict" in checkpoint:
            self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.g_optimizer.load_state_dict(checkpoint["g_optimizer"])
        self.d_optimizer.load_state_dict(checkpoint["d_optimizer"])
        if (
            self.g_lr_scheduler is not None
            and checkpoint.get("g_lr_scheduler") is not None
        ):
            self.g_lr_scheduler.load_state_dict(checkpoint["g_lr_scheduler"])
        if (
            self.d_lr_scheduler is not None
            and checkpoint.get("d_lr_scheduler") is not None
        ):
            self.d_lr_scheduler.load_state_dict(checkpoint["d_lr_scheduler"])
        if self.use_amp and checkpoint.get("g_scaler") is not None:
            self.g_scaler.load_state_dict(checkpoint["g_scaler"])
        if self.use_amp and checkpoint.get("d_scaler") is not None:
            self.d_scaler.load_state_dict(checkpoint["d_scaler"])
        self.logger.info(
            f"Checkpoint loaded. Resume training from epoch {self.start_epoch}"
        )
