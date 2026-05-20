import csv
import warnings
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
import torchaudio
from tqdm.auto import tqdm
from transformers import AutoFeatureExtractor, XcodecModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", category=UserWarning)


def read_lj_metadata(data_root):
    rows = []
    with open(data_root / "metadata.csv", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE)
        for row in reader:
            if len(row) < 3:
                continue
            utt_id, _, normalized_text = row[0], row[1], row[2]
            rows.append((utt_id, normalized_text.strip()))
    return rows


@hydra.main(version_base=None, config_path="../src/configs", config_name="tts_bonus")
def main(config):
    tts_cfg = config.tts
    data_root = Path(tts_cfg.data_root)
    out_dir = Path(tts_cfg.codes_cache)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading X-Codec on {device}...")
    model = XcodecModel.from_pretrained(tts_cfg.xcodec_model_id).to(device).eval()
    feat = AutoFeatureExtractor.from_pretrained(tts_cfg.xcodec_model_id)
    target_sr = feat.sampling_rate

    rows = read_lj_metadata(data_root)
    out_meta = []
    with torch.no_grad():
        for utt_id, text in tqdm(rows, desc="extract codes"):
            wav_path = data_root / "wavs" / f"{utt_id}.wav"
            if not wav_path.exists():
                continue
            wav, sr = torchaudio.load(str(wav_path))
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != target_sr:
                wav = torchaudio.functional.resample(wav, sr, target_sr)
            audio_np = wav.squeeze(0).numpy()
            inputs = feat(raw_audio=audio_np, sampling_rate=target_sr, return_tensors="pt").to(device)
            codes = model.encode(inputs["input_values"], bandwidth=tts_cfg.bandwidth, return_dict=False)
            codes_np = codes[0, 0].cpu().numpy().astype(np.int16)
            np.save(out_dir / f"{utt_id}.npy", codes_np)
            out_meta.append((utt_id, text, int(codes_np.shape[0])))

    print(f"saved {len(out_meta)} utterances")
    with open(out_dir / "metadata.csv", "w", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\")
        for utt_id, text, n in out_meta:
            w.writerow([utt_id, text, n])
    print(f"metadata: {out_dir / 'metadata.csv'}")


if __name__ == "__main__":
    main()
