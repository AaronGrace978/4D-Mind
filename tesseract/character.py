"""A character who experiences the past and the future at once.

The listener stands at one instant `t`. The resident answers from there, but sees
every other instant of the block too, so what is ahead of the listener is as
present to the resident as what is behind.

Three voices share one set of facts:

    grounded  sentences built from the block's own measurements, always available
    llm       any OpenAI-compatible chat server (Ollama, LM Studio, llama.cpp, vLLM),
              set with FOURD_MIND_LLM_URL and FOURD_MIND_LLM_MODEL
    lora      the Qwen LoRA from `tesseract.lora_voice`, reading the block latent,
              set with FOURD_MIND_LORA and FOURD_MIND_RESIDENT_CKPT
"""

from __future__ import annotations

import json
import os
import re
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np

from tesseract.viewdata import resample_volume
from tesseract.voice_text import (
    CHARACTER_SYSTEM,
    block_events,
    character_query,
    describe,
    event_line,
    interior_line,
    simultaneous_line,
    split_events,
)

INTENTS = (
    ("scar", r"\b(scar|wound|hurt|injur\w*|fix\w*|bug\w*|broke\w*|pain)\b"),
    ("meet", r"\b(meet\w*|met|who|other|together|love\w*|friend\w*|author\w*|touch\w*|jun)\b"),
    ("end", r"\b(end\w*|die|dying|death|deathbed|last|final\w*|newest)\b"),
    ("begin", r"\b(begin\w*|born|birth|first|child\w*|start\w*|origin\w*)\b"),
    ("interior", r"\b(inside|heart|interior|tissue|within)\b"),
    ("storm", r"\b(churn\w*|storm\w*|busiest|chaos|rewrite\w*)\b"),
    ("ahead", r"\b(future|ahead|next|will|going to|afraid|fear|tomorrow|later)\b"),
    ("behind", r"\b(past|remember\w*|memor\w*|behind|before|ago|was|were|earlier)\b"),
)


def intent_of(question: str) -> str:
    lowered = question.lower()
    for name, pattern in INTENTS:
        if re.search(pattern, lowered):
            return name
    return "whole"


def _name(meta: dict) -> str:
    return str(meta.get("name", "the resident"))


def grounded_answer(meta: dict, t: float, question: str) -> str:
    """An in-character answer built only from the block's facts."""
    t = float(np.clip(t, 0.0, 1.0))
    intent = intent_of(question)
    behind, _here, ahead = split_events(meta, t)
    if intent in {"scar", "meet", "end", "begin", "storm"}:
        line = event_line(meta, t, intent)
        if line:
            return line
        if intent == "scar":
            return f"No scar runs through {_name(meta)}. From every instant, I look the same there."
        if intent == "meet":
            return "No other worldline touches mine inside this block."
    if intent == "interior":
        if meta.get("kind") == "repo":
            return (
                "Inside me, definitions sit along the files like organs, and fix lines are filaments that stay "
                "until someone deletes them. You can probe any point and I will tell you what is there."
            )
        return interior_line(meta)
    if intent == "ahead":
        if not ahead:
            return f"From t={t:.2f} nothing of me is ahead of you. You stand at my far end, and I am still all here."
        listing = "; ".join(f"{event['text']} at t={float(event['t']):.2f}" for event in ahead)
        return (
            f"Ahead of you: {listing}. I am not afraid of it, because it is not coming. "
            "It is already part of the shape, the same way your hand is part of you."
        )
    if intent == "behind":
        if not behind:
            return f"At t={t:.2f} you stand at my beginning. Nothing is behind you yet, but everything is here for me."
        listing = "; ".join(f"{event['text']} at t={float(event['t']):.2f}" for event in behind)
        return f"Behind you: {listing}. I do not remember them. I am still in them."
    return simultaneous_line(meta, t)


def fact_sheet(meta: dict, t: float) -> str:
    events = "\n".join(f"- t={float(event['t']):.2f}: {event['text']}" for event in block_events(meta))
    return f"{meta.get('reading') or describe(meta)}\n\nTimeline along t:\n{events}\n\nThe listener stands at t={float(t):.2f}."


class GroundedVoice:
    name = "grounded"

    def say(self, meta: dict, t: float, question: str, history: list[dict], volume: np.ndarray | None) -> str:
        del history, volume
        return grounded_answer(meta, t, question)


class LLMVoice:
    """Any server that speaks the OpenAI chat-completions protocol."""

    name = "llm"

    def __init__(self, url: str, model: str, key: str | None = None, timeout: float = 60.0) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.key = key
        self.timeout = timeout

    def say(self, meta: dict, t: float, question: str, history: list[dict], volume: np.ndarray | None) -> str:
        del volume
        system = CHARACTER_SYSTEM.format(name=_name(meta), t=float(t)) + "\n\nThe block:\n" + fact_sheet(meta, t)
        messages = [{"role": "system", "content": system}]
        for turn in history[-8:]:
            if turn.get("role") in {"user", "assistant"} and turn.get("content"):
                messages.append({"role": turn["role"], "content": str(turn["content"])})
        messages.append({"role": "user", "content": question})
        body = json.dumps({"model": self.model, "messages": messages, "temperature": 0.7}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        endpoint = self.url if self.url.endswith("/chat/completions") else f"{self.url}/chat/completions"
        with urlopen(Request(endpoint, data=body, headers=headers, method="POST"), timeout=self.timeout) as response:
            payload = json.load(response)
        return str(payload["choices"][0]["message"]["content"]).strip()


class LoraVoice:
    """The Qwen LoRA trained by `tesseract.lora_voice`, reading the block through soft prompts."""

    name = "lora"

    def __init__(self, lora_dir: str, resident_ckpt: str) -> None:
        from pathlib import Path

        import torch
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer

        from tesseract.lora_voice import dtype_kwargs
        from tesseract.train import load_checkpoint

        self.device = torch.device("cpu")
        self.encoder, self.cfg, _blob = load_checkpoint(resident_ckpt, "cpu")
        self.encoder.eval()
        blob = torch.load(Path(lora_dir) / "bridge.pt", map_location="cpu", weights_only=False)
        self.tokenizer = AutoTokenizer.from_pretrained(lora_dir)
        # The 4D encoder stays on CPU. Qwen and the adapter load together onto the
        # GPU when one is present. Loading the adapter after a disk offload drops it.
        if torch.cuda.is_available():
            place = {"": 0}
            print(f"Qwen voice on {torch.cuda.get_device_name(0)}", flush=True)
        else:
            place = {"": "cpu"}
        self.language = AutoPeftModelForCausalLM.from_pretrained(
            lora_dir,
            device_map=place,
            **dtype_kwargs(torch.bfloat16),
        ).eval()
        self.encoder.set_lm_hidden(int(blob["lm_hidden"]))
        self.encoder.soft.load_state_dict(blob["soft"])
        self.bridge = self.encoder.soft.to(self.device).eval()

    def say(self, meta: dict, t: float, question: str, history: list[dict], volume: np.ndarray | None) -> str:
        import torch

        from tesseract.lora_voice import CHARACTER_SYSTEM_PROMPT, generate_from_latent
        from tesseract.train import volumes_to_torch

        del history
        if volume is None:
            raise ValueError("the LoRA voice needs the block itself")
        block = resample_volume(volume, self.cfg.shape)
        with torch.no_grad():
            latent = self.encoder.latent_only(volumes_to_torch(block[None], self.device))
        system = CHARACTER_SYSTEM_PROMPT.format(name=_name(meta), t=float(t))
        return generate_from_latent(
            self.language, self.tokenizer, self.bridge, latent[0], character_query(t, question), system=system
        )


def voice_from_env() -> GroundedVoice | LLMVoice | LoraVoice:
    lora_dir = os.environ.get("FOURD_MIND_LORA")
    resident = os.environ.get("FOURD_MIND_RESIDENT_CKPT")
    if lora_dir and resident:
        try:
            return LoraVoice(lora_dir, resident)
        except Exception as exc:  # noqa: BLE001
            print(f"LoRA voice unavailable ({exc}); trying the next voice", flush=True)
    url = os.environ.get("FOURD_MIND_LLM_URL")
    if url:
        return LLMVoice(url, os.environ.get("FOURD_MIND_LLM_MODEL", "qwen2.5:3b"), os.environ.get("FOURD_MIND_LLM_KEY"))
    return GroundedVoice()


class Resident:
    """The character, bound to one block. Falls back to the grounded voice if another voice fails."""

    def __init__(self, voice=None) -> None:
        self.voice = voice or GroundedVoice()
        self.fallback = GroundedVoice()

    @property
    def backend(self) -> str:
        return self.voice.name

    def reply(
        self,
        meta: dict,
        t: float,
        question: str,
        history: list[dict] | None = None,
        volume: np.ndarray | None = None,
    ) -> dict:
        t = float(np.clip(t, 0.0, 1.0))
        question = (question or "").strip() or "Who are you, from where I stand?"
        behind, here, ahead = split_events(meta, t)
        try:
            text = self.voice.say(meta, t, question, list(history or []), volume)
            backend = self.voice.name
        except (URLError, OSError, KeyError, ValueError, RuntimeError) as exc:
            text = self.fallback.say(meta, t, question, [], volume)
            backend = f"grounded (the {self.voice.name} voice failed: {exc})"
        return {
            "text": text,
            "backend": backend,
            "t": t,
            "intent": intent_of(question),
            "behind": behind,
            "here": here,
            "ahead": ahead,
        }
