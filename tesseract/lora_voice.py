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

from tesseract.character import grounded_answer
from tesseract.codeblock import SCULPTURES, ambient_sculpture, rasterize_repo
from tesseract.coder import SYSTEM as CODER_SYSTEM, fourd_code_lens, grounded_code_review
from tesseract.life import load_iris, rasterize, repo_root
from tesseract.risk import report as risk_report
from tesseract.train import load_checkpoint, make_cache, pick_device, volumes_to_torch
from tesseract.voice_text import CHARACTER_QUESTIONS, CHARACTER_SYSTEM, QUERY_TEXT, SYSTEM, answer_for, character_query

CHARACTER_SYSTEM_PROMPT = CHARACTER_SYSTEM
CODE_QUESTIONS = (
    "Where is the next bug likely to land in this 4D code solid, and why?",
    "Describe the 4D spatiotemporal lens of this codebase.",
    "Which files co-mutate along time, and where do fix filaments persist?",
    "What is the highest-risk hotspot in this solid, and how should it be patched?",
)


def code_examples(
    shape: tuple[int, int, int, int],
    rng: np.random.Generator,
    n_sculptures: int = 4,
    include_self_repo: bool = True,
) -> tuple[list[np.ndarray], list[dict], list[tuple[int, str, str, str]]]:
    """Build 4D code solids and `(solid_offset, system, query, answer)` rows for LoRA coding."""
    volumes: list[np.ndarray] = []
    metas: list[dict] = []
    rows: list[tuple[int, str, str, str]] = []
    root = repo_root()
    if include_self_repo and (root / ".git").exists():
        try:
            vol, _inv, meta = rasterize_repo(root, shape, max_commits=120)
            volumes.append(vol)
            metas.append(meta)
        except Exception:  # noqa: BLE001
            pass
    sculpture_keys = list(SCULPTURES)
    for idx in range(n_sculptures):
        kind = sculpture_keys[idx % len(sculpture_keys)]
        vol, _inv, meta = ambient_sculpture(kind, shape, seed=int(rng.integers(0, 10_000)))
        volumes.append(vol)
        metas.append(meta)
    for local_idx, (vol, meta) in enumerate(zip(volumes, metas)):
        raw = risk_report(vol, meta, None, top=6)
        spots = raw.get("hotspots") or []
        lens = fourd_code_lens(vol, meta, spots)
        src_root = root if meta.get("kind") == "repo" else None
        for question in CODE_QUESTIONS:
            answer, _files = grounded_code_review(meta, src_root, question, lens, spots, None)
            rows.append((local_idx, CODER_SYSTEM, question, answer))
        for _ in range(2):
            t = float(np.round(rng.uniform(0.1, 0.95), 2))
            q = CHARACTER_QUESTIONS[int(rng.integers(0, len(CHARACTER_QUESTIONS)))]
            sys_prompt = CHARACTER_SYSTEM.format(name=meta.get("name", "the code solid"), t=t)
            rows.append((local_idx, sys_prompt, character_query(t, q), grounded_answer(meta, t, q)))
    return volumes, metas, rows


def character_examples(metas: list[dict], per_life: int, rng: np.random.Generator) -> list[tuple[int, str, str, str]]:
    """`(life, system, query, answer)` rows where the listener stands at a random instant.

    The answer depends on where the listener stands, so the voice has to place
    every event of the solid relative to that instant at once.
    """
    rows = []
    for index, meta in enumerate(metas):
        for _ in range(per_life):
            t = float(np.round(rng.uniform(0.0, 1.0), 2))
            question = CHARACTER_QUESTIONS[int(rng.integers(0, len(CHARACTER_QUESTIONS)))]
            system = CHARACTER_SYSTEM.format(name=meta.get("name", "the resident"), t=t)
            rows.append((index, system, character_query(t, question), grounded_answer(meta, t, question)))
    return rows


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


def labels_from_prompt_lens(
    input_ids: torch.Tensor, prompt_lens: list[int], attention: torch.Tensor
) -> torch.Tensor:
    """Train on the answer only. Padding is found from `attention`, not a pad id,
    because some tokenizers pad with the same id that ends the answer."""
    labels = input_ids.clone()
    for row, length in enumerate(prompt_lens):
        labels[row, : int(length)] = -100
    labels[attention == 0] = -100
    return labels


def _token_ids(tokenizer, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _chat_ids(tokenizer, query: str, answer: str | None, system: str = SYSTEM) -> tuple[list[int], int]:
    # Render to text first: transformers 5 returns a dict from apply_chat_template(tokenize=True).
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]
    prompt = _token_ids(tokenizer, tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))
    if answer is None:
        return prompt, len(prompt)
    full = _token_ids(
        tokenizer,
        tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": answer}],
            tokenize=False,
        ),
    )
    if full[: len(prompt)] == prompt:
        return full, len(prompt)
    return full, max(len(full) - len(_token_ids(tokenizer, answer)), 0)


def dtype_kwargs(dtype: torch.dtype) -> dict:
    """`dtype=` on transformers 4.56 and later, `torch_dtype=` before that."""
    import transformers

    major, minor = (int(part) for part in transformers.__version__.split(".")[:2])
    return {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}


def match_embedding_scale(bridge: torch.nn.Linear, language) -> None:
    """Start soft prompts at the size of real token embeddings.

    A default init makes them far larger than any word the language model has
    seen, and the first steps spend themselves shrinking the bridge.
    """
    with torch.no_grad():
        target = float(language.get_input_embeddings().weight.float().std())
        torch.nn.init.normal_(bridge.weight, std=target / max(bridge.in_features, 1) ** 0.5)
        torch.nn.init.zeros_(bridge.bias)


def stop_ids(tokenizer) -> set[int]:
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    for token in ("<|im_end|>", "<|endoftext|>"):
        found = tokenizer.convert_tokens_to_ids(token)
        if isinstance(found, int) and found != tokenizer.unk_token_id and found >= 0:
            ids.add(found)
    return ids


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
    language = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", **dtype_kwargs(torch.bfloat16))
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
    character_per_life: int = 6,
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
    rng = np.random.default_rng(seed)
    code_vols, _code_metas, code_rows = code_examples(
        cfg.shape, rng, n_sculptures=max(2, min(4, cache // 2)), include_self_repo=True
    )
    code_offset = len(volumes)
    volumes.extend(code_vols)
    print(f"encoding {len(volumes)} solids (lives + 4D code blocks) into block latents", flush=True)
    latents = precompute_latents(encoder, volumes, device, batch_size=max(1, min(2, lm_batch)))
    encoder.to("cpu")
    language, tokenizer = _load_language_model(model_name)
    encoder.set_lm_hidden(int(language.config.hidden_size))
    bridge = encoder.soft.to(device)
    match_embedding_scale(bridge, language)
    bridge.train()
    plain = [(index, key) for index in range(len(metas)) for key in QUERY_TEXT]
    plain.extend((0, key) for _ in range(3) for key in QUERY_TEXT)
    examples = [(index, SYSTEM, QUERY_TEXT[key], answer_for(metas[index], key)) for index, key in plain]
    examples.extend(character_examples(metas, character_per_life, rng))
    examples.extend((code_offset + idx, sys_p, q, ans) for idx, sys_p, q, ans in code_rows)
    opt = torch.optim.AdamW(
        [
            {"params": bridge.parameters(), "lr": lr_bridge},
            {"params": [p for p in language.parameters() if p.requires_grad], "lr": lr_lora},
        ]
    )
    language.train()
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    for step in range(1, steps + 1):
        pick = rng.integers(0, len(examples), size=lm_batch)
        ids_batch = []
        prompt_lens = []
        latent_rows = []
        for choice in pick:
            index, system, query, answer = examples[int(choice)]
            ids, plen = _chat_ids(tokenizer, query, answer, system)
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
        labels = labels_from_prompt_lens(input_ids, prompt_lens, attn)
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
        json.dumps(
            {
                "base": model_name,
                "steps": steps,
                "queries": QUERY_TEXT,
                "character_questions": list(CHARACTER_QUESTIONS),
                "character_per_life": character_per_life,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}", flush=True)
    return out


@torch.no_grad()
def generate_from_latent(
    language,
    tokenizer,
    bridge,
    latent: torch.Tensor,
    query: str,
    max_new_tokens: int = 160,
    system: str = SYSTEM,
) -> str:
    language.eval()
    bridge.eval()
    ids, _plen = _chat_ids(tokenizer, query, None, system)
    lm_device = _lm_device(language)
    prompt = torch.tensor(ids, dtype=torch.long, device=lm_device).view(1, -1)
    embeds = language.get_input_embeddings()(prompt)
    hidden = embeds.shape[-1]
    n_soft = bridge.out_features // hidden
    soft = bridge(latent.view(1, -1).to(device=bridge.weight.device, dtype=bridge.weight.dtype))
    soft = soft.view(1, n_soft, hidden).to(device=embeds.device, dtype=embeds.dtype)
    inputs = torch.cat([soft, embeds], dim=1)
    attention = torch.ones(inputs.shape[:2], dtype=torch.long, device=embeds.device)
    stops = stop_ids(tokenizer)
    made: list[int] = []
    past = None
    # One new token per step. Replaying the whole prompt each time does not fit
    # beside a 7B model on a 16GB card.
    for _ in range(max_new_tokens):
        out = language(
            inputs_embeds=inputs,
            attention_mask=attention,
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values
        nxt = int(out.logits[:, -1].argmax(dim=-1).item())
        if nxt in stops:
            break
        made.append(nxt)
        inputs = language.get_input_embeddings()(torch.tensor([[nxt]], device=embeds.device))
        attention = torch.cat(
            [attention, torch.ones((1, 1), dtype=torch.long, device=embeds.device)],
            dim=1,
        )
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
    parser.add_argument("--character-per-life", type=int, default=6, help="Character turns per life, from random instants")
    args = parser.parse_args()
    train_voice(
        args.encoder,
        args.out,
        args.model,
        steps=args.steps,
        device=pick_device(args.device),
        cache=args.cache,
        lm_batch=args.batch,
        character_per_life=args.character_per_life,
    )


if __name__ == "__main__":
    main()
