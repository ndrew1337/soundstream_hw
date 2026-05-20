import math
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

def _init_audio_embeddings_from_codebook(emb_weight, codebook_vectors, audio_offset, seed=0):
    n_codes, codec_dim = codebook_vectors.shape
    lm_dim = emb_weight.shape[1]
    g = torch.Generator().manual_seed(seed)
    a = math.sqrt(6.0 / (codec_dim + lm_dim))
    proj = torch.empty(codec_dim, lm_dim).uniform_(-a, a, generator=g)
    projected = codebook_vectors.float() @ proj
    text_norms = emb_weight[:audio_offset].float().norm(dim=1)
    target_norm = text_norms.mean()
    projected_norms = projected.norm(dim=1, keepdim=True).clamp_min(1e-6)
    projected = projected * (target_norm / projected_norms)
    emb_weight[audio_offset:audio_offset + n_codes].copy_(projected.to(emb_weight.dtype))


def build_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg["lm_model_id"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    special = {
        "additional_special_tokens": [cfg["audio_start_token"], cfg["audio_end_token"]],
    }
    tokenizer.add_special_tokens(special)
    audio_start_id = tokenizer.convert_tokens_to_ids(cfg["audio_start_token"])
    audio_end_id = tokenizer.convert_tokens_to_ids(cfg["audio_end_token"])
    audio_offset = len(tokenizer)
    return tokenizer, audio_start_id, audio_end_id, audio_offset


def build_model(cfg, tokenizer, audio_offset, codebook_size, device, xcodec_model=None):
    base = AutoModelForCausalLM.from_pretrained(cfg["lm_model_id"], dtype=torch.float32)
    new_vocab_size = audio_offset + codebook_size
    base.resize_token_embeddings(new_vocab_size)

    init_mode = cfg.get("audio_embed_init", "mean")
    with torch.no_grad():
        emb = base.get_input_embeddings().weight
        old_size = audio_offset
        if init_mode == "codebook":
            if xcodec_model is None:
                raise ValueError("audio_embed_init='codebook' requires xcodec_model")
            cb = xcodec_model.quantizer.quantizers[0].codebook.embed.detach().cpu()
            _init_audio_embeddings_from_codebook(
                emb.data, cb, old_size, seed=int(cfg.get("audio_embed_init_seed", 0))
            )
        else:
            mean = emb[:old_size].mean(0, keepdim=True)
            std = emb[:old_size].std(0, keepdim=True).clamp_min(1e-6)
            emb[old_size:].copy_(mean + 0.02 * std * torch.randn_like(emb[old_size:]))
        out_emb = base.get_output_embeddings().weight
        if out_emb.data_ptr() != emb.data_ptr():
            out_emb[old_size:].copy_(emb[old_size:])

    mode = cfg.get("mode", "lora")
    if mode == "lora":

        lora_cfg = LoraConfig(
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            target_modules=list(cfg["lora_target_modules"]),
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=["embed_tokens", "lm_head"],
        )
        base = get_peft_model(base, lora_cfg)
        base.print_trainable_parameters()
    elif mode == "lm_head":
        for p in base.parameters():
            p.requires_grad = False
        for p in base.get_input_embeddings().parameters():
            p.requires_grad = True
        out_emb = base.get_output_embeddings()
        if out_emb is not None:
            for p in out_emb.parameters():
                p.requires_grad = True
    elif mode == "full":
        pass
    else:
        raise ValueError(f"unknown mode: {mode}")

    return base.to(device)


def resume_lora_model(cfg, audio_offset, codebook_size, device, adapter_path):
    base = AutoModelForCausalLM.from_pretrained(cfg["lm_model_id"], dtype=torch.float32)
    base.resize_token_embeddings(audio_offset + codebook_size)
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
    model.print_trainable_parameters()
    return model.to(device)


def freeze_text_embeddings(model, audio_offset):
    """Keep text-token embedding rows fixed; audio/code rows still update."""

    def _zero_text_grad(grad):
        if grad is None:
            return grad
        grad = grad.clone()
        grad[:audio_offset] = 0
        return grad

    embed = model.get_input_embeddings()
    embed.weight.register_hook(_zero_text_grad)

    out_emb = model.get_output_embeddings()
    if out_emb is not None and hasattr(out_emb, "weight"):
        if out_emb.weight.data_ptr() != embed.weight.data_ptr():
            out_emb.weight.register_hook(_zero_text_grad)


def load_for_inference(cfg, ckpt_path, device):
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    audio_start_id = tokenizer.convert_tokens_to_ids(cfg["audio_start_token"])
    audio_end_id = tokenizer.convert_tokens_to_ids(cfg["audio_end_token"])
    audio_offset = len(tokenizer) - cfg["codebook_size"]

    adapter_cfg = Path(ckpt_path) / "adapter_config.json"
    if adapter_cfg.exists():
        base = AutoModelForCausalLM.from_pretrained(cfg["lm_model_id"], dtype=torch.float32)
        base.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(base, ckpt_path)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(ckpt_path, dtype=torch.float32)
    model.eval().to(device)
    return tokenizer, model, audio_start_id, audio_end_id, audio_offset
