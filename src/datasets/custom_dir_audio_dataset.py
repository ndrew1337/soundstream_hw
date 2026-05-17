from pathlib import Path

import torchaudio

from src.datasets.base_audio_dataset import BaseAudioDataset


class CustomDirAudioDataset(BaseAudioDataset):
    AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus"}

    def __init__(self, audio_dir: str, *args, **kwargs):
        index = []
        for path in sorted(Path(audio_dir).iterdir()):
            if path.suffix.lower() not in self.AUDIO_EXTS:
                continue
            t_info = torchaudio.info(str(path))
            index.append(
                {
                    "path": str(path.absolute().resolve()),
                    "audio_len": t_info.num_frames / t_info.sample_rate,
                }
            )
        super().__init__(index, *args, **kwargs)
