import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def read_codes_metadata(metadata_csv):
    rows = []
    with open(metadata_csv, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\")
        for row in reader:
            if len(row) < 3:
                continue
            utt_id, text, n = row[0], row[1], int(row[2])
            rows.append((utt_id, text, n))
    return rows


def split_train_test(rows, holdout_size):
    train = rows[:-holdout_size]
    test = rows[-holdout_size:]
    return train, test


class LJSpeechCodes(Dataset):
    def __init__(
        self,
        rows,
        codes_cache,
        tokenizer,
        audio_start_id,
        audio_end_id,
        audio_offset,
        codebook_size,
        max_seq_len=1024,
    ):
        self.rows = rows
        self.codes_cache = Path(codes_cache)
        self.tokenizer = tokenizer
        self.audio_start_id = audio_start_id
        self.audio_end_id = audio_end_id
        self.audio_offset = audio_offset
        self.codebook_size = codebook_size
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        utt_id, text, _ = self.rows[idx]
        codes = np.load(self.codes_cache / f"{utt_id}.npy").astype(np.int64)
        codes = np.clip(codes, 0, self.codebook_size - 1)
        audio_ids = codes + self.audio_offset

        text_ids = self.tokenizer.encode(text, add_special_tokens=False)
        bos = self.tokenizer.bos_token_id
        prefix = ([bos] if bos is not None else []) + text_ids + [self.audio_start_id]
        suffix = audio_ids.tolist() + [self.audio_end_id]
        ids = prefix + suffix
        ids = ids[: self.max_seq_len]

        prefix_len = min(len(prefix), len(ids))
        labels = list(ids)
        for i in range(prefix_len):
            labels[i] = -100
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(pad_token_id):
    def _collate(batch):
        max_len = max(item["input_ids"].size(0) for item in batch)
        input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, item in enumerate(batch):
            L = item["input_ids"].size(0)
            input_ids[i, :L] = item["input_ids"]
            labels[i, :L] = item["labels"]
            attn[i, :L] = 1
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}

    return _collate
