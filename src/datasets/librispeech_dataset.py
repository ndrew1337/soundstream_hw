import json
import os
import shutil
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from src.datasets.base_audio_dataset import BaseAudioDataset
from src.utils.audio_io import audio_duration_sec
from src.utils.io_utils import ROOT_PATH

URL_LINKS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
}


class LibrispeechDataset(BaseAudioDataset):
    def __init__(
        self,
        part: str,
        data_dir: Optional[str] = None,
        *args,
        **kwargs,
    ):
        assert part in URL_LINKS or part == "train_all"

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "librispeech"
        data_dir = Path(data_dir)
        data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir

        if part == "train_all":
            index = sum(
                [self._get_or_load_index(p) for p in URL_LINKS if "train" in p],
                [],
            )
        else:
            index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _resolve_part_dir(self, part: str) -> Optional[Path]:
        candidates = [
            self._data_dir / part,
            self._data_dir / "LibriSpeech" / part,
        ]
        for cand in candidates:
            if cand.exists() and any(cand.iterdir()):
                return cand
        return None

    def _load_part(self, part: str) -> Path:
        import wget

        arch_path = self._data_dir / f"{part}.tar.gz"
        if not arch_path.exists():
            print(f"Downloading LibriSpeech partition {part}...")
            wget.download(URL_LINKS[part], str(arch_path))
            print()

        print(f"Extracting {arch_path.name}...")
        shutil.unpack_archive(arch_path, self._data_dir)
        os.remove(str(arch_path))

        part_dir = self._resolve_part_dir(part)
        if part_dir is None:
            raise RuntimeError(
                f"Failed to locate {part} after extraction in {self._data_dir}."
            )
        return part_dir

    def _get_or_load_index(self, part: str) -> list:
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                return json.load(f)

        index = self._create_index(part)
        with index_path.open("w") as f:
            json.dump(index, f, indent=2)
        return index

    def _create_index(self, part: str) -> list:
        part_dir = self._resolve_part_dir(part)
        if part_dir is None:
            part_dir = self._load_part(part)

        flac_dirs = set()
        for dirpath, _dirnames, filenames in os.walk(str(part_dir)):
            if any(f.endswith(".flac") for f in filenames):
                flac_dirs.add(dirpath)

        index = []
        for flac_dir in tqdm(
            sorted(flac_dirs), desc=f"Indexing LibriSpeech {part}"
        ):
            flac_dir = Path(flac_dir)
            for flac_path in sorted(flac_dir.glob("*.flac")):
                length = audio_duration_sec(flac_path)
                index.append(
                    {
                        "path": str(flac_path.absolute().resolve()),
                        "audio_len": length,
                    }
                )
        return index
