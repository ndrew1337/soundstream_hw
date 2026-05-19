import logging
import random
from typing import List, Optional

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from src.utils.audio_io import load_audio_mono

logger = logging.getLogger(__name__)


class BaseAudioDataset(Dataset):
    def __init__(
        self,
        index: List[dict],
        target_sr: int = 16000,
        limit: Optional[int] = None,
        max_audio_length: Optional[float] = None,
        min_audio_length: Optional[float] = None,
        shuffle_index: bool = False,
        sort_by_length: bool = False,
        instance_transforms: Optional[dict] = None,
    ):
        self._assert_index_is_valid(index)

        index = self._filter_records_from_dataset(
            index, min_audio_length, max_audio_length
        )
        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        if sort_by_length and not shuffle_index:
            index = self._sort_index(index)

        self._index: List[dict] = index
        self.target_sr = target_sr
        self.instance_transforms = instance_transforms

    def __len__(self):
        return len(self._index)

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        audio_path = data_dict["path"]
        audio = self.load_audio(audio_path)

        instance_data = {
            "audio": audio,
            "audio_len": audio.shape[-1],
            "audio_path": audio_path,
        }
        instance_data = self.preprocess_data(instance_data)
        instance_data["audio_len"] = instance_data["audio"].shape[-1]
        return instance_data

    def load_audio(self, path: str) -> torch.Tensor:
        audio_tensor, sr = load_audio_mono(path)
        if sr != self.target_sr:
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, sr, self.target_sr
            )
        return audio_tensor

    def preprocess_data(self, instance_data: dict) -> dict:
        if self.instance_transforms is None:
            return instance_data
        for transform_name, transform in self.instance_transforms.items():
            if transform is None:
                continue
            instance_data[transform_name] = transform(instance_data[transform_name])
        return instance_data

    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "path" in entry
            assert "audio_len" in entry

    @staticmethod
    def _filter_records_from_dataset(
        index: list,
        min_audio_length: Optional[float],
        max_audio_length: Optional[float],
    ) -> list:
        if min_audio_length is None and max_audio_length is None:
            return index

        lens = np.array([el["audio_len"] for el in index], dtype=np.float64)
        keep = np.ones_like(lens, dtype=bool)
        if min_audio_length is not None:
            keep &= lens >= min_audio_length
        if max_audio_length is not None:
            keep &= lens <= max_audio_length

        dropped = (~keep).sum()
        if dropped:
            logger.info(
                "Filtered %d (%.1f%%) records by audio length [%s, %s] sec.",
                dropped,
                100.0 * dropped / len(lens),
                min_audio_length,
                max_audio_length,
            )
        return [el for el, k in zip(index, keep) if k]

    @staticmethod
    def _sort_index(index):
        return sorted(index, key=lambda x: x["audio_len"])

    @staticmethod
    def _shuffle_and_limit_index(index, limit, shuffle_index):
        if shuffle_index:
            random.seed(42)
            random.shuffle(index)
        if limit is not None:
            index = index[:limit]
        return index
