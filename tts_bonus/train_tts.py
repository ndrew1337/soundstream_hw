import math
import re
import time
import warnings
import sys
from pathlib import Path

import hydra
import numpy as np
import soundfile as sf
import torch
import torchaudio
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pystoi import stoi as stoi_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoFeatureExtractor, XcodecModel
from torchmetrics.audio.nisqa import NonIntrusiveSpeechQualityAssessment

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate_tts import synthesize
from src.datasets.ljspeech_codes import (
    LJSpeechCodes,
    collate_fn,
    read_codes_metadata,
    split_train_test,
)
from src.model.codec_lm import build_model, build_tokenizer, resume_lora_model, freeze_text_embeddings
from src.utils.init_utils import set_random_seed, setup_saving_and_logging
from src.utils.io_utils import ROOT_PATH
from src.metrics.tracker import MetricTracker

warnings.filterwarnings("ignore", category=UserWarning)


def load_wav(path):
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(audio.T), int(sr)




def masked_audio_ce_loss(logits, labels, audio_offset, codebook_size, audio_end_id):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    mask = shift_labels != -100
    if not mask.any():
        return logits.sum() * 0.0
    flat_logits = shift_logits[mask]
    flat_labels = shift_labels[mask]
    allowed = torch.zeros(flat_logits.shape[-1], dtype=torch.bool, device=flat_logits.device)
    allowed[audio_offset:audio_offset + codebook_size] = True
    allowed[audio_end_id] = True
    flat_logits[:, ~allowed] = float("-inf")
    return torch.nn.functional.cross_entropy(flat_logits, flat_labels)

def cosine_lr(step, warmup, total, base_lr):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * min(1.0, p)))


def training_lr(step, base_lr, warmup, max_steps, start_step=0, prev_max_steps=None):
    if start_step <= 0 or prev_max_steps is None or prev_max_steps == max_steps:
        return cosine_lr(step, warmup, max_steps, base_lr)
    if step < start_step:
        return cosine_lr(step, warmup, prev_max_steps, base_lr)
    lr_resume = cosine_lr(start_step - 1, warmup, prev_max_steps, base_lr)
    if max_steps <= start_step:
        return lr_resume
    progress = (step - start_step) / (max_steps - start_step)
    return lr_resume * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def evaluate(model, loader, device, audio_offset=None, codebook_size=None, audio_end_id=None):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    use_masked = audio_offset is not None and codebook_size is not None and audio_end_id is not None
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            labels = batch["labels"]
            mask = labels != -100
            n = mask.sum().item()
            if use_masked:
                loss_val = masked_audio_ce_loss(
                    out.logits, labels, audio_offset, codebook_size, audio_end_id,
                )
                total_loss += loss_val.item() * n
            else:
                total_loss += out.loss.item() * n
            shift_logits = out.logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            shift_mask = shift_labels != -100
            if use_masked:
                allowed_eval = torch.zeros(shift_logits.shape[-1], dtype=torch.bool, device=shift_logits.device)
                allowed_eval[audio_offset:audio_offset + codebook_size] = True
                allowed_eval[audio_end_id] = True
                shift_logits_m = shift_logits.clone()
                shift_logits_m[..., ~allowed_eval] = float("-inf")
                preds = shift_logits_m.argmax(dim=-1)
            else:
                preds = shift_logits.argmax(dim=-1)
            total_correct += ((preds == shift_labels) & shift_mask).sum().item()
            total_tokens += n
    model.train()
    loss = total_loss / max(1, total_tokens)
    accuracy = total_correct / max(1, total_tokens)
    return loss, accuracy


def audio_evaluate(
    model,
    xcodec,
    tokenizer,
    test_rows,
    data_root,
    sr,
    tts_cfg,
    audio_start_id,
    audio_end_id,
    audio_offset,
    codebook_size,
    device,
    n_samples,
    writer,
    step,
    nisqa=None,
    n_log_samples=2,
    seed=0,
):
    model.eval()
    stois, nisqas = [], []
    n_success = 0
    rows = test_rows[:n_samples]
    log_ids = {utt_id for utt_id, _, _ in rows[:n_log_samples]}
    for utt_id, text, _ in rows:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        gen_wav = synthesize(
            text,
            tokenizer,
            model,
            xcodec,
            audio_start_id,
            audio_end_id,
            audio_offset,
            codebook_size,
            device,
            max_new_tokens=tts_cfg["generate_max_new_tokens"],
            temperature=tts_cfg["generate_temperature"],
            top_p=tts_cfg["generate_top_p"],
        )
        if gen_wav is None:
            continue
        n_success += 1
        ref_wav, ref_sr = load_wav(data_root / f"{utt_id}.wav")
        if ref_wav.shape[0] > 1:
            ref_wav = ref_wav.mean(0, keepdim=True)
        if ref_sr != sr:
            ref_wav = torchaudio.functional.resample(ref_wav, ref_sr, sr)
        ref_wav = ref_wav.squeeze(0)
        gen_wav_s = gen_wav.squeeze(0) if gen_wav.ndim > 1 else gen_wav
        target_len = ref_wav.shape[-1]
        if gen_wav_s.shape[-1] != target_len and gen_wav_s.shape[-1] > 0:
            gen_stretched = torch.nn.functional.interpolate(
                gen_wav_s.unsqueeze(0).unsqueeze(0).float(),
                size=target_len,
                mode="linear",
                align_corners=False,
            ).squeeze()
        else:
            gen_stretched = gen_wav_s.float()
        stois.append(float(stoi_metric(ref_wav.numpy(), gen_stretched.numpy(), sr, extended=False)))
        if nisqa is not None:
            try:
                nisqas.append(float(nisqa(gen_wav.to(device)).item()))
            except Exception:
                pass
        if utt_id in log_ids:
            writer.set_step(step, mode="test")
            writer.add_audio(f"gen_{utt_id}", gen_wav_s.cpu(), sr, caption=text)
    model.train()
    stoi_mean = float(np.mean(stois)) if stois else 0.0
    nisqa_mean = float(np.mean(nisqas)) if nisqas else None
    return stoi_mean, nisqa_mean, n_success


@hydra.main(version_base=None, config_path="../src/configs", config_name="tts_bonus")
def main(config):
    set_random_seed(config.trainer.seed)

    save_dir = ROOT_PATH / config.trainer.save_dir / config.writer.run_name
    resume_from = config.trainer.get("resume_from")
    start_step = 0
    prev_max_steps = None
    if resume_from:
        if (save_dir / "config.yaml").exists():
            old_cfg = OmegaConf.load(save_dir / "config.yaml")
            prev_max_steps = int(old_cfg.trainer.max_steps)
        m = re.search(r"checkpoint-(\d+)", str(resume_from))
        start_step = int(m.group(1)) if m else 0

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    tts_cfg = OmegaConf.to_container(config.tts, resolve=True)
    tokenizer, audio_start_id, audio_end_id, audio_offset = build_tokenizer(tts_cfg)

    audio_n_eval_samples = int(config.trainer.get("audio_n_eval_samples", 0))
    init_mode = tts_cfg.get("audio_embed_init", "mean")
    need_xcodec_for_init = init_mode == "codebook" and not resume_from
    need_xcodec_for_eval = audio_n_eval_samples > 0
    xcodec = None
    xcodec_sr = None
    if need_xcodec_for_init or need_xcodec_for_eval:
        logger.info("Loading X-Codec...")
        xcodec = XcodecModel.from_pretrained(tts_cfg["xcodec_model_id"]).to(device).eval()
        feat = AutoFeatureExtractor.from_pretrained(tts_cfg["xcodec_model_id"])
        xcodec_sr = feat.sampling_rate

    if resume_from:
        resume_path = save_dir / resume_from
        model = resume_lora_model(
            tts_cfg, audio_offset, tts_cfg["codebook_size"], device, str(resume_path)
        )
        freeze_text_embeddings(model, audio_offset)
        lr_resume = training_lr(
            start_step,
            config.trainer.lr,
            config.trainer.warmup_steps,
            config.trainer.max_steps,
            start_step,
            prev_max_steps,
        )
        optimizer_state_path = resume_path / "training_state.pt"
        if optimizer_state_path.exists():
            training_state = torch.load(str(optimizer_state_path), map_location=device)
            logger.info("restoring optimizer state from checkpoint")
        else:
            training_state = None
        logger.info(f"resumed from {resume_path} at step {start_step}")
        logger.info(
            f"lr schedule: {lr_resume:.2e} at resume "
            f"(prev max_steps {prev_max_steps}, max_steps {config.trainer.max_steps})"
        )
    else:
        training_state = None
        logger.info(f"audio embedding init: {init_mode}")
        model = build_model(
            tts_cfg, tokenizer, audio_offset, tts_cfg["codebook_size"], device,
            xcodec_model=xcodec,
        )
        freeze_text_embeddings(model, audio_offset)
    logger.info(f"vocab size: {len(tokenizer)}, audio_offset: {audio_offset}")

    rows = read_codes_metadata(tts_cfg["metadata_csv"])
    train_rows, test_rows = split_train_test(rows, tts_cfg["holdout_size"])
    logger.info(f"train: {len(train_rows)}, holdout: {len(test_rows)}")

    train_ds = LJSpeechCodes(
        train_rows,
        tts_cfg["codes_cache"],
        tokenizer,
        audio_start_id,
        audio_end_id,
        audio_offset,
        tts_cfg["codebook_size"],
        config.trainer.max_seq_len,
    )
    test_ds = LJSpeechCodes(
        test_rows,
        tts_cfg["codes_cache"],
        tokenizer,
        audio_start_id,
        audio_end_id,
        audio_offset,
        tts_cfg["codebook_size"],
        config.trainer.max_seq_len,
    )
    coll = collate_fn(tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.trainer.batch_size,
        shuffle=True,
        num_workers=config.trainer.num_workers,
        collate_fn=coll,
        drop_last=True,
        pin_memory=(device == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.trainer.batch_size,
        shuffle=False,
        num_workers=config.trainer.num_workers,
        collate_fn=coll,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.trainer.lr, weight_decay=config.trainer.weight_decay)
    if resume_from and training_state is not None:
        optimizer.load_state_dict(training_state["optimizer"])
        logger.info("optimizer state restored")
    use_bf16 = device == "cuda"

    audio_n_log_samples = int(config.trainer.get("audio_n_log_samples", 2))
    audio_eval_seed = int(config.trainer.get("audio_eval_seed", 0))
    nisqa = None
    data_root_wavs = None
    if audio_n_eval_samples > 0:
        data_root_wavs = Path(tts_cfg["data_root"]) / "wavs"
        try:
            nisqa = NonIntrusiveSpeechQualityAssessment(xcodec_sr).to(device)
        except Exception as e:
            logger.info(f"NISQA unavailable: {e}")

    model.train()
    epoch = 0
    step = start_step
    accum = 0
    train_metrics = MetricTracker("loss", "perplexity", "grad_norm", "learning rate", "throughput_tps", writer=writer)
    last_log_time = time.time()
    optimizer.zero_grad()
    pbar = tqdm(total=config.trainer.max_steps, initial=start_step, desc="train")
    train_iter = iter(train_loader)
    while step < config.trainer.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if use_bf16
            else torch.amp.autocast("cpu", enabled=False)
        )
        with ctx:
            out = model(**batch)
            loss_val = masked_audio_ce_loss(
                out.logits, batch["labels"],
                audio_offset, tts_cfg["codebook_size"], audio_end_id,
            )
            loss = loss_val / config.trainer.grad_accum_steps
        loss.backward()
        n = (batch["labels"] != -100).sum().item()
        train_metrics.update("loss", loss_val.item(), n)
        accum += 1
        if accum >= config.trainer.grad_accum_steps:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, config.trainer.clip_grad_norm).item()
            train_metrics.update("grad_norm", grad_norm)
            lr_now = training_lr(
                step,
                config.trainer.lr,
                config.trainer.warmup_steps,
                config.trainer.max_steps,
                start_step,
                prev_max_steps,
            )
            for g in optimizer.param_groups:
                g["lr"] = lr_now
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            accum = 0
            pbar.update(1)
            if step % config.trainer.log_step == 0:
                avg = train_metrics.avg("loss")
                ppl = math.exp(min(avg, 30.0))
                now = time.time()
                elapsed = max(1e-6, now - last_log_time)
                tokens_total = train_metrics._data.loc["loss", "counts"] if "loss" in train_metrics._data.index else 0
                tps = tokens_total / elapsed
                train_metrics.update("perplexity", ppl)
                train_metrics.update("learning rate", lr_now)
                train_metrics.update("throughput_tps", tps)
                pbar.set_postfix(loss=f"{avg:.4f}", ppl=f"{ppl:.1f}", lr=f"{lr_now:.2e}")
                writer.set_step(step)
                for key, val in train_metrics.result().items():
                    writer.add_scalar(key, val)
                train_metrics.reset()
                last_log_time = now
            if step % config.trainer.save_step == 0 or step == config.trainer.max_steps:
                ckpt_dir = save_dir / f"checkpoint-{step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                if hasattr(model, "save_pretrained"):
                    model.save_pretrained(str(ckpt_dir))
                else:
                    torch.save(model.state_dict(), ckpt_dir / "pytorch_model.bin")
                tokenizer.save_pretrained(str(ckpt_dir))
                torch.save({
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "epoch": epoch,
                    "config": OmegaConf.to_container(config),
                }, ckpt_dir / "training_state.pt")
                logger.info(f"Saving checkpoint: {ckpt_dir} ...")
            if step % config.trainer.eval_step == 0:
                eval_loss, eval_acc = evaluate(model, test_loader, device, audio_offset, tts_cfg["codebook_size"], audio_end_id)
                eval_ppl = math.exp(min(eval_loss, 30.0))
                logger.info(
                    f"step {step} eval loss: {eval_loss:.4f} ppl: {eval_ppl:.2f} acc: {eval_acc:.4f}"
                )
                writer.set_step(step, mode="test")
                writer.add_scalar("loss", eval_loss)
                writer.add_scalar("perplexity", eval_ppl)
                writer.add_scalar("token_accuracy", eval_acc)
                if xcodec is not None:
                    stoi_mean, nisqa_mean, n_success = audio_evaluate(
                        model,
                        xcodec,
                        tokenizer,
                        test_rows,
                        data_root_wavs,
                        xcodec_sr,
                        tts_cfg,
                        audio_start_id,
                        audio_end_id,
                        audio_offset,
                        tts_cfg["codebook_size"],
                        device,
                        audio_n_eval_samples,
                        writer,
                        step,
                        nisqa=nisqa,
                        n_log_samples=audio_n_log_samples,
                        seed=audio_eval_seed,
                    )
                    msg = f"step {step} STOI: {stoi_mean:.4f} (n_ok {n_success}/{audio_n_eval_samples})"
                    if nisqa_mean is not None:
                        msg += f" NISQA: {nisqa_mean:.4f}"
                    logger.info(msg)
                    writer.set_step(step, mode="test")
                    writer.add_scalar("STOI", stoi_mean)
                    writer.add_scalar("audio_n_success", n_success)
                    if nisqa_mean is not None:
                        writer.add_scalar("NISQA", nisqa_mean)
            if step == config.trainer.max_steps and xcodec is not None and step % config.trainer.eval_step != 0:
                logger.info(f"final audio eval at step {step}...")
                stoi_mean, nisqa_mean, n_success = audio_evaluate(
                    model,
                    xcodec,
                    tokenizer,
                    test_rows,
                    data_root_wavs,
                    xcodec_sr,
                    tts_cfg,
                    audio_start_id,
                    audio_end_id,
                    audio_offset,
                    tts_cfg["codebook_size"],
                    device,
                    audio_n_eval_samples,
                    writer,
                    step,
                    nisqa=nisqa,
                    n_log_samples=audio_n_log_samples,
                    seed=audio_eval_seed,
                )
                msg = f"final STOI: {stoi_mean:.4f} (n_ok {n_success}/{audio_n_eval_samples})"
                if nisqa_mean is not None:
                    msg += f" NISQA: {nisqa_mean:.4f}"
                logger.info(msg)
                writer.set_step(step, mode="test")
                writer.add_scalar("STOI", stoi_mean)
                writer.add_scalar("audio_n_success", n_success)
                if nisqa_mean is not None:
                    writer.add_scalar("NISQA", nisqa_mean)
    pbar.close()


if __name__ == "__main__":
    main()
