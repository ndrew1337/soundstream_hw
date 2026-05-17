import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torchaudio
from hydra import compose, initialize
from hydra.utils import instantiate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--checkpoint", default="saved/baseline/model_best.pth")
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with initialize(config_path="../src/configs", version_base=None):
        cfg = compose(config_name="inference")

    model = instantiate(cfg.model).to(device).eval()
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state.get("state_dict", state))

    audio, sr = torchaudio.load(args.input)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != args.sample_rate:
        audio = torchaudio.functional.resample(audio, sr, args.sample_rate)

    with torch.no_grad():
        x_fake, _, _ = model(audio.unsqueeze(0).to(device))
    x_fake = x_fake.squeeze(0).cpu()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(args.output, x_fake, args.sample_rate)


if __name__ == "__main__":
    main()
