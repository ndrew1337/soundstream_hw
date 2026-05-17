from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio


def _to_1d_tensor(audio: torch.Tensor) -> torch.Tensor:
    audio = audio.detach().cpu()
    while audio.dim() > 1:
        audio = audio.squeeze(0)
    return audio


def mel_spectrogram(
    audio: torch.Tensor,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
    f_min: float = 80.0,
    f_max: Optional[float] = None,
    eps: float = 1e-5,
) -> np.ndarray:
    wave = _to_1d_tensor(audio).float().unsqueeze(0)
    hi = float(f_max or (sample_rate / 2.0))
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        n_mels=n_mels,
        power=1.0,
        f_min=f_min,
        f_max=hi,
    )(wave)
    log_mel = torch.log(mel + eps).squeeze(0).numpy()
    return log_mel


def _figure_to_numpy(fig: plt.Figure, dpi: int = 140) -> np.ndarray:
    fig.set_dpi(dpi)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buf = np.asarray(fig.canvas.buffer_rgba()).reshape(height, width, 4)
    return buf[..., :3].copy()


def render_mel_image(
    mel: np.ndarray,
    title: Optional[str] = None,
    diverging: bool = False,
    sample_rate: int = 16000,
    hop_length: int = 256,
    figsize_width: float = 12.0,
    figsize_height: float = 4.0,
    dpi: int = 140,
    mel_interpolation: str = "bilinear",
    robust_pct: Tuple[float, float] = (2.0, 98.0),
) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))
    interp = mel_interpolation
    use_robust = robust_pct is not None and not diverging
    if diverging:
        vmax = float(np.nanmax(np.abs(mel))) or 1e-6
        im = ax.imshow(
            mel,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation=interp,
            resample=False,
        )
    elif use_robust:
        lo, hi = np.percentile(mel, robust_pct)
        if hi - lo < 1e-6:
            lo, hi = float(mel.min()), float(mel.max()) + 1e-6
        im = ax.imshow(
            mel,
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=lo,
            vmax=hi,
            interpolation=interp,
            resample=False,
        )
    else:
        im = ax.imshow(
            mel,
            origin="lower",
            aspect="auto",
            cmap="magma",
            interpolation=interp,
            resample=False,
        )
    duration = mel.shape[1] * hop_length / sample_rate
    ax.set_xlabel(f"frames (~{duration:.3f}s)")
    ax.set_ylabel("mel bin")
    if title is not None:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    img = _figure_to_numpy(fig, dpi=dpi)
    plt.close(fig)
    return img


def render_waveform_image(
    audio: torch.Tensor,
    sample_rate: int = 16000,
    title: Optional[str] = None,
    color: str = "tab:blue",
) -> np.ndarray:
    wave = _to_1d_tensor(audio).float().numpy()
    times = np.arange(wave.shape[0]) / sample_rate
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot(times, wave, color=color, linewidth=0.6)
    ax.set_xlim(times[0], times[-1] if times.size > 0 else 0.0)
    ax.set_xlabel("time, s")
    ax.set_ylabel("amplitude")
    ax.grid(True, alpha=0.3)
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    img = _figure_to_numpy(fig)
    plt.close(fig)
    return img


def compute_pair_visuals(
    x_real: torch.Tensor,
    x_fake: torch.Tensor,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
    mel_f_min: float = 80.0,
    mel_f_max: Optional[float] = None,
    mel_robust_pct: Tuple[float, float] = (2.0, 98.0),
    mel_dpi: int = 140,
    mel_fig_width: float = 12.0,
    mel_fig_height: float = 4.0,
    mel_interpolation: str = "bilinear",
) -> dict:
    wave_real = _to_1d_tensor(x_real).float()
    wave_fake = _to_1d_tensor(x_fake).float()
    length = min(wave_real.shape[-1], wave_fake.shape[-1])
    wave_real = wave_real[:length]
    wave_fake = wave_fake[:length]
    wave_diff = wave_fake - wave_real

    mel_real = mel_spectrogram(
        wave_real, sample_rate, n_fft, hop_length, n_mels, f_min=mel_f_min, f_max=mel_f_max
    )
    mel_fake = mel_spectrogram(
        wave_fake, sample_rate, n_fft, hop_length, n_mels, f_min=mel_f_min, f_max=mel_f_max
    )
    mel_diff = mel_fake - mel_real

    return {
        "wave_real": wave_real,
        "wave_fake": wave_fake,
        "wave_diff": wave_diff,
        "mel_real": mel_real,
        "mel_fake": mel_fake,
        "mel_diff": mel_diff,
        "mel_real_image": render_mel_image(
            mel_real,
            title="mel: input",
            sample_rate=sample_rate,
            hop_length=hop_length,
            dpi=mel_dpi,
            figsize_width=mel_fig_width,
            figsize_height=mel_fig_height,
            robust_pct=mel_robust_pct,
            mel_interpolation=mel_interpolation,
        ),
        "mel_fake_image": render_mel_image(
            mel_fake,
            title="mel: reconstruction",
            sample_rate=sample_rate,
            hop_length=hop_length,
            dpi=mel_dpi,
            figsize_width=mel_fig_width,
            figsize_height=mel_fig_height,
            robust_pct=mel_robust_pct,
            mel_interpolation=mel_interpolation,
        ),
        "mel_diff_image": render_mel_image(
            mel_diff,
            title="mel: difference",
            diverging=True,
            sample_rate=sample_rate,
            hop_length=hop_length,
            dpi=mel_dpi,
            figsize_width=mel_fig_width,
            figsize_height=mel_fig_height,
            robust_pct=None,
            mel_interpolation=mel_interpolation,
        ),
    }


def save_pair_artifacts(
    out_dir: Path,
    sample_id: int,
    visuals: dict,
    sample_rate: int = 16000,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"sample_{sample_id:06d}"
    torchaudio.save(str(out_dir / f"{prefix}_wave_input.wav"), visuals["wave_real"].unsqueeze(0), sample_rate)
    torchaudio.save(str(out_dir / f"{prefix}_wave_reconstruction.wav"), visuals["wave_fake"].unsqueeze(0), sample_rate)
    torchaudio.save(str(out_dir / f"{prefix}_wave_difference.wav"), visuals["wave_diff"].unsqueeze(0), sample_rate)
    plt.imsave(str(out_dir / f"{prefix}_mel_input.png"), visuals["mel_real_image"])
    plt.imsave(str(out_dir / f"{prefix}_mel_reconstruction.png"), visuals["mel_fake_image"])
    plt.imsave(str(out_dir / f"{prefix}_mel_difference.png"), visuals["mel_diff_image"])
