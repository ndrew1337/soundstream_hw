import csv
import warnings
import sys
from pathlib import Path

import hydra
import numpy as np
import soundfile as sf
import torch
import torchaudio
from omegaconf import OmegaConf
from pystoi import stoi as stoi_metric
from tqdm.auto import tqdm
from transformers import AutoFeatureExtractor, XcodecModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate_tts import synthesize
from src.datasets.ljspeech_codes import read_codes_metadata, split_train_test
from src.model.codec_lm import load_for_inference
from src.utils.init_utils import set_random_seed
from torchmetrics.audio.nisqa import NonIntrusiveSpeechQualityAssessment

warnings.filterwarnings("ignore", category=UserWarning)


def load_wav(path):
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(audio.T), int(sr)


def time_stretch_to_length(wav, target_len):
    src_len = wav.shape[-1]
    if src_len == target_len or src_len == 0:
        return wav
    x = wav.unsqueeze(0) if wav.ndim == 1 else wav.unsqueeze(0)
    x = torch.nn.functional.interpolate(x, size=target_len, mode="linear", align_corners=False)
    return x.squeeze(0)


@hydra.main(version_base=None, config_path="../src/configs", config_name="tts_bonus")
def main(config):
    set_random_seed(config.trainer.seed)
    if not config.ckpt:
        raise ValueError("provide ckpt=<path> on the command line")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts_cfg = OmegaConf.to_container(config.tts, resolve=True)
    tokenizer, model, audio_start_id, audio_end_id, audio_offset = load_for_inference(
        tts_cfg, config.ckpt, device
    )
    xcodec = XcodecModel.from_pretrained(tts_cfg["xcodec_model_id"]).to(device).eval()
    feat = AutoFeatureExtractor.from_pretrained(tts_cfg["xcodec_model_id"])
    sr = feat.sampling_rate

    rows = read_codes_metadata(tts_cfg["metadata_csv"])
    _, test_rows = split_train_test(rows, tts_cfg["holdout_size"])
    test_rows = test_rows[: tts_cfg["n_eval_samples"]]

    nisqa = None
    nisqa = NonIntrusiveSpeechQualityAssessment(sr).to(device)

    out_path = Path(tts_cfg["eval_out_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio_dir = Path(tts_cfg["eval_audio_dir"]) if tts_cfg["eval_audio_dir"] else None
    if audio_dir is not None:
        audio_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(tts_cfg["data_root"]) / "wavs"
    stois, nisqas = [], []
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "text", "stoi", "nisqa"])
        for utt_id, text, _ in tqdm(test_rows, desc="eval"):
            gen_wav = synthesize(
                text,
                tokenizer,
                model,
                xcodec,
                audio_start_id,
                audio_end_id,
                audio_offset,
                tts_cfg["codebook_size"],
                device,
                max_new_tokens=tts_cfg["generate_max_new_tokens"],
                temperature=tts_cfg["generate_temperature"],
                top_p=tts_cfg["generate_top_p"],
            )
            if gen_wav is None:
                w.writerow([utt_id, text, "", ""])
                continue
            ref_wav, ref_sr = load_wav(data_root / f"{utt_id}.wav")
            if ref_wav.shape[0] > 1:
                ref_wav = ref_wav.mean(0, keepdim=True)
            if ref_sr != sr:
                ref_wav = torchaudio.functional.resample(ref_wav, ref_sr, sr)
            ref_wav = ref_wav.squeeze(0)
            gen_wav = gen_wav.squeeze(0) if gen_wav.ndim > 1 else gen_wav
            target_len = ref_wav.shape[-1]
            gen_stretched = time_stretch_to_length(gen_wav, target_len)
            stoi_val = float(
                stoi_metric(ref_wav.numpy(), gen_stretched.numpy(), sr, extended=False)
            )
            nisqa_val = ""
            if nisqa is not None:
                try:
                    nisqa_val = float(nisqa(gen_wav.to(device)).item())
                except Exception:
                    nisqa_val = ""
            stois.append(stoi_val)
            if nisqa_val != "":
                nisqas.append(nisqa_val)
            w.writerow([utt_id, text, f"{stoi_val:.4f}", f"{nisqa_val}"])
            if audio_dir is not None:
                torchaudio.save(str(audio_dir / f"{utt_id}_gen.wav"), gen_wav.unsqueeze(0), sr)

    logs = {"test_STOI": np.mean(stois)}
    if nisqas:
        logs["test_NISQA"] = np.mean(nisqas)
    for key, value in logs.items():
        print(f"    {key:15s}: {value}")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
