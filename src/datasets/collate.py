from typing import List

import torch


def collate_fn(dataset_items: List[dict]) -> dict:
    if not dataset_items:
        raise ValueError("collate_fn received an empty batch.")

    audios = [el["audio"] for el in dataset_items]
    lens = torch.tensor(
        [el["audio_len"] for el in dataset_items], dtype=torch.long
    )

    max_len = int(lens.max().item())
    batch_size = len(audios)
    channels = audios[0].shape[0]

    padded = torch.zeros(batch_size, channels, max_len, dtype=audios[0].dtype)
    for i, a in enumerate(audios):
        padded[i, :, : a.shape[-1]] = a

    return {
        "audio": padded,
        "audio_len": lens,
        "audio_path": [el["audio_path"] for el in dataset_items],
    }
