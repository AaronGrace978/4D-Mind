"""LoRA-tune a language model whose first tokens are the block latent.

The encoder stays frozen. A small bridge turns its latent into soft prompts,
and LoRA adapts Qwen so those continuous tokens are a sense organ.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tesseract.life import load_iris, rasterize
from tesseract.train import load_checkpoint, make_cache, pick_device, volumes_to_torch
from tesseract.voice_text import QUERY_TEXT, SYSTEM, answer_for


def _lm_device(language) -> torch.device:
    return language.get_input_embeddings().weight.device


def attach_soft_prompts(soft, token_embeds, attention_mask, labels):
    """Prepend soft prompts. Their labels are ignored; text labels pass through."""
    batch, prompts, _hidden = soft.shape
    inputs = torch.cat(
        [soft.to(dtype=token_embeds.dtype, device=token_embeds.device), token_embeds],
        dim=1,
    )
    soft_attn = torch.ones(batch, prompts, dtype=attention_mask.dtype, device=attention_mask.device)
    attention = torch.cat([soft_attn, attention_mask], dim=1)
    soft_labels = torch.full((batch, prompts), -100, dtype=labels.dtype, device=labels.device)
    full_labels = torch.cat([soft_labels, labels], dim=1)
    return inputs, attention, full_labels


def labels_from_prompt_lens(input_ids: torch.Tensor, prompt_lens: list[int], pad_id: int) -> torch.Tensor:
    labels = input_ids.clone()
    for row, length in enumerate(prompt_lens):
        labels[row, : int(length)] = -100
    labels[input_ids == pad_id] = -100
    return labels


def _chat_ids(tokenizer, query: str, answer: str | None) -> tuple[list[int], int]:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": query},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    if answer is None:
        return list(prompt), len(prompt)
    full = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": answer}],
        tokenize=True,
    )
    prompt = list(prompt)
    full = list(full)
    if full[: len(prompt)] == prompt:
        return full, len(prompt)
    answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
    return full, max(len(full) - len(answer_ids), 0)


def precompute_latents(model, volumes: list[np.ndarray], device, batch_size: int = 2) -> torch.Tensor:
    model.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(volumes), batch_size):
            chunk = np.stack(volumes[start : start + batch_size])
            tensor = volumes_to_torch(chunk, device)
            rows.append(model.latent_only(tensor).float().cpu())
    return torch.cat(rows, dim=0)


def _load_language_model(model_name: str):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    language = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    language = get_peft_model(
        language,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    language.print_trainable_parameters()
    return language, tokenizer


def train_voice(
    encoder_ckpt: str,
    out_dir: str,
    model_name: str,
    steps: int,
    device,
    cache: int = 16,
    lm_batch: int = 1,
    lr_bridge: float = 1e-3,
    lr_lora: float = 1e-4,
    seed: int = 7,
    log_every: int = 20,
) -> Path:
    device = torch.device(device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    encoder, cfg, _blob = load_checkpoint(encoder_ckpt, device)
    volumes, _inv, metas = make_cache(cfg, cache, seed, mode="procedural")
    iris_volume, iris_inv, iris_meta = rasterize(load_iris(), cfg.shape)
    volumes[0] = iris_volume
    metas[0] = iris_meta
    metas[0]["invariants"] = iris_inv.tolist()
    print(f"encoding {len(volumes)} lives into block latents", flush=True)
    latents = precompute_latents(encoder, volumes, device, batch_size=max(1, min(2, lm_batch)))
    encoder.to("cpu")
    language, tokenizer = _load_language_model(model_name)
    encoder.set_lm_hidden(int(language.config.hidden_size))
    bridge = encoder.soft.to(device)
    bridge.train()
    examples = [(index, key) for index in range(len(metas)) for key in QUERY_TEXT]
    examples.extend((0, key) for _ in range(3) for key in QUERY_TEXT)
    opt = torch.optim.AdamW(
        [
            {"params": bridge.parameters(), "lr": lr_bridge},
            {"params": [p for p in language.parameters() if p.requires_grad], "lr": lr_lora},
        ]
    )
    rng = np.random.default_rng(seed)
    language.train()
    pad_id = int(tokenizer.pad_token_id)
    for step in range(1, steps + 1):
        pick = rng.integers(0, len(examples), size=lm_batch)
        ids_batch = []
        prompt_lens = []
        latent_rows = []
        for choice in pick:
            index, key = examples[int(choice)]
            ids, plen = _chat_ids(tokenizer, QUERY_TEXT[key], answer_for(metas[index], key))
            ids_batch.append(ids)
            prompt_lens.append(plen)
            latent_rows.append(latents[index])
        width = max(len(row) for row in ids_batch)
        input_ids = torch.full((lm_batch, width), pad_id, dtype=torch.long)
        attn = torch.zeros(lm_batch, width, dtype=torch.long)
        for row, ids in enumerate(ids_batch):
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn[row, : len(ids)] = 1
        lm_device = _lm_device(language)
        input_ids = input_ids.to(lm_device)
        attn = attn.to(lm_device)
        labels = labels_from_prompt_lens(input_ids, prompt_lens, pad_id)
        latent = torch.stack(latent_rows).to(device=bridge.weight.device, dtype=bridge.weight.dtype)
        soft = bridge(latent).view(lm_batch, cfg.n_soft_prompts, encoder.cfg.lm_hidden)
        token_embeds = language.get_input_embeddings()(input_ids)
        inputs, attention, full_labels = attach_soft_prompts(soft, token_embeds, attn, labels)
        opt.zero_grad(set_to_none=True)
        loss = language(inputs_embeds=inputs, attention_mask=attention, labels=full_labels).loss
        loss.backward()
        opt.step()
        if log_every and (step % log_every == 0 or step == 1 or step == steps):
            print(f"lora step {step:5d}  loss {float(loss.detach()):.4f}", flush=True)
    language.save_pretrained(out)
    tokenizer.save_pretrained(out)
    torch.save(
        {
            "soft": {key: value.detach().cpu() for key, value in bridge.state_dict().items()},
            "lm_hidden": int(encoder.cfg.lm_hidden),
            "n_soft_prompts": int(cfg.n_soft_prompts),
            "dim": int(cfg.dim),
            "base": model_name,
        },
        out / "bridge.pt",
    )
    (out / "voice.json").write_text(
        json.dumps({"base": model_name, "steps": steps, "queries": QUERY_TEXT}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out}", flush=True)
    return out


@torch.no_grad()
def generate_from_latent(language, tokenizer, bridge, latent: torch.Tensor, query: str, max_new_tokens: int = 160) -> str:
    language.eval()
    bridge.eval()
    ids, _plen = _chat_ids(tokenizer, query, None)
    lm_device = _lm_device(language)
    prompt = torch.tensor(ids, dtype=torch.long, device=lm_device).view(1, -1)
    embeds = language.get_input_embeddings()(prompt)
    hidden = embeds.shape[-1]
    n_soft = bridge.out_features // hidden
    soft = bridge(latent.view(1, -1).to(device=bridge.weight.device, dtype=bridge.weight.dtype))
    soft = soft.view(1, n_soft, hidden).to(device=embeds.device, dtype=embeds.dtype)
    inputs = torch.cat([soft, embeds], dim=1)
    attention = torch.ones(inputs.shape[:2], dtype=torch.long, device=embeds.device)
    eos = tokenizer.eos_token_id
    made: list[int] = []
    for _ in range(max_new_tokens):
        logits = language(inputs_embeds=inputs, attention_mask=attention).logits[:, -1]
        nxt = int(logits.argmax(dim=-1).item())
        made.append(nxt)
        if eos is not None and nxt == int(eos):
            break
        nxt_embed = language.get_input_embeddings()(torch.tensor([[nxt]], device=embeds.device))
        inputs = torch.cat([inputs, nxt_embed], dim=1)
        attention = torch.ones(inputs.shape[:2], dtype=torch.long, device=embeds.device)
    return tokenizer.decode(made, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA-tune Qwen on block-latent soft prompts")
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--out", default="checkpoints/qwen_block_lora")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cache", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    train_voice(
        args.encoder,
        args.out,
        args.model,
        steps=args.steps,
        device=pick_device(args.device),
        cache=args.cache,
        lm_batch=args.batch,
    )


if __name__ == "__main__":
    main()
