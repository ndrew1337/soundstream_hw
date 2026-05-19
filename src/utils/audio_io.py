from pathlib import Path

import soundfile as sf
import torch


def audio_duration_sec(path: str | Path) -> float:
    info = sf.info(str(path))
    return info.frames / float(info.samplerate)


def load_audio_mono(path: str | Path) -> tuple[torch.Tensor, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.shape[1] > 1:
        audio = audio[:, :1]
    return torch.from_numpy(audio.T.copy()), int(sr)
