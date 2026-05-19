from pathlib import Path

from src.datasets.base_audio_dataset import BaseAudioDataset
from src.utils.audio_io import audio_duration_sec


class CustomDirAudioDataset(BaseAudioDataset):
    AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus"}

    def __init__(self, audio_dir: str, *args, **kwargs):
        index = []
        for path in sorted(Path(audio_dir).iterdir()):
            if path.suffix.lower() not in self.AUDIO_EXTS:
                continue
            index.append(
                {
                    "path": str(path.absolute().resolve()),
                    "audio_len": audio_duration_sec(path),
                }
            )
        super().__init__(index, *args, **kwargs)
