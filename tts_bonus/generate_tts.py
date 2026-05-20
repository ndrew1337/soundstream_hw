import warnings
import sys
from pathlib import Path

import hydra
import torch
import torchaudio
from omegaconf import OmegaConf
from transformers import AutoFeatureExtractor, XcodecModel, LogitsProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.codec_lm import load_for_inference

warnings.filterwarnings("ignore", category=UserWarning)

class AudioTokenLogitsProcessor(LogitsProcessor):
    def __init__(self, audio_offset, codebook_size, audio_end_id):
        self.allowed = None
        self.audio_offset = audio_offset
        self.codebook_size = codebook_size
        self.audio_end_id = audio_end_id

    def __call__(self, input_ids, scores):
        if self.allowed is None:
            self.allowed = torch.zeros(scores.shape[-1], dtype=torch.bool, device=scores.device)
            self.allowed[self.audio_offset:self.audio_offset + self.codebook_size] = True
            self.allowed[self.audio_end_id] = True
        mask = torch.where(self.allowed, 0.0, float("-inf"))
        return scores + mask.unsqueeze(0)

def synthesize(
    text,
    tokenizer,
    model,
    xcodec,
    audio_start_id,
    audio_end_id,
    audio_offset,
    codebook_size,
    device,
    max_new_tokens=600,
    temperature=0.8,
    top_p=0.9,
):
    text_ids = tokenizer.encode(text, add_special_tokens=False)
    bos = tokenizer.bos_token_id
    prefix = ([bos] if bos is not None else []) + text_ids + [audio_start_id]
    input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
    attn = torch.ones_like(input_ids)
    do_sample = temperature is not None and temperature > 0
    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        eos_token_id=audio_end_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=top_p)
    processor = AudioTokenLogitsProcessor(audio_offset, codebook_size, audio_end_id)
    gen_kwargs["logits_processor"] = [processor]
    with torch.no_grad():
        out = model.generate(**gen_kwargs)
    generated = out[0, input_ids.size(1):].cpu()
    generated = generated[generated != audio_end_id]
    audio_ids = generated[(generated >= audio_offset) & (generated < audio_offset + codebook_size)]
    codes = (audio_ids - audio_offset).clamp(0, codebook_size - 1)
    if codes.numel() == 0:
        return None
    codes = codes.view(1, 1, -1).to(device)
    with torch.no_grad():
        wav = xcodec.decode(codes).audio_values
    return wav.squeeze(0).cpu()


@hydra.main(version_base=None, config_path="../src/configs", config_name="tts_bonus")
def main(config):
    if not config.ckpt or not config.text:
        raise ValueError("provide ckpt=<path> and text='...' on the command line")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts_cfg = OmegaConf.to_container(config.tts, resolve=True)
    tokenizer, model, audio_start_id, audio_end_id, audio_offset = load_for_inference(
        tts_cfg, config.ckpt, device
    )
    xcodec = XcodecModel.from_pretrained(tts_cfg["xcodec_model_id"]).to(device).eval()
    feat = AutoFeatureExtractor.from_pretrained(tts_cfg["xcodec_model_id"])
    sr = feat.sampling_rate

    wav = synthesize(
        config.text,
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
    if wav is None:
        print("no audio tokens generated")
        return
    torchaudio.save(config.out, wav, sr)
    print(f"saved {config.out} ({wav.shape[-1] / sr:.2f}s)")


if __name__ == "__main__":
    main()
