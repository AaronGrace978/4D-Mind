"""Ollama Cloud (https://ollama.com). The model list is the live catalog.

Chat needs an API key in the `Authorization: Bearer` header. The key is read
from `OLLAMA_API_KEY` or from the coder panel for this process only.
"""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

CLOUD_API = "https://ollama.com/api"

# Used when ollama.com cannot be reached. The running app prefers /api/tags.
FALLBACK_MODELS = (
    "kimi-k2.7-code",
    "deepseek-v4-pro:0813",
    "deepseek-v4.1-flash",
    "gpt-oss:120b",
    "gpt-oss:20b",
    "glm-5.3",
    "glm-5.3-flash",
    "glm-5.2",
    "gemma4:31b",
    "kimi-k3",
    "kimi-k2.6",
    "minimax-m3",
    "minimax-m2.7",
    "mistral-large-4",
    "mistral-large-3:675b",
    "nemotron-3-ultra",
    "nemotron-3-super",
    "nemotron-3-nano:30b",
)


def is_coder(name: str) -> bool:
    lowered = name.lower()
    return "code" in lowered or lowered.startswith("gpt-oss") or lowered.startswith("deepseek")


def order_models(names: list[str]) -> list[dict]:
    rows = [{"name": name, "coder": is_coder(name)} for name in dict.fromkeys(names) if name]
    rows.sort(key=lambda row: (not row["coder"], row["name"]))
    return rows


def default_model(models: list[dict]) -> str | None:
    for row in models:
        if "code" in row["name"].lower():
            return row["name"]
    return models[0]["name"] if models else None


def list_models(opener=urlopen, timeout: float = 20) -> list[dict]:
    """Every model currently published on Ollama Cloud, coding models first."""
    request = Request(f"{CLOUD_API}/tags", headers={"Accept": "application/json"})
    with opener(request, timeout=timeout) as response:
        payload = json.load(response)
    names = []
    for item in payload.get("models") or []:
        name = item.get("name") or item.get("model")
        if name:
            names.append(str(name))
    if not names:
        raise ValueError("Ollama Cloud returned no models")
    return order_models(names)


def fallback_models() -> list[dict]:
    return order_models(list(FALLBACK_MODELS))


def api_key(explicit: str | None = None) -> str:
    return (explicit or os.environ.get("OLLAMA_API_KEY") or "").strip()


def chat(model: str, messages: list[dict], key: str, opener=urlopen, timeout: float = 180) -> str:
    """One non-streaming Ollama Cloud chat completion."""
    if not key:
        raise RuntimeError("Set an Ollama Cloud API key. Create one under Settings → API keys on ollama.com.")
    if not model:
        raise RuntimeError("Choose a cloud model.")
    body = json.dumps({"model": model, "messages": messages, "stream": False}).encode("utf-8")
    request = Request(
        f"{CLOUD_API}/chat",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with opener(request, timeout=timeout) as response:
            payload = json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise RuntimeError(f"Ollama Cloud returned {exc.code}. {detail}".strip()) from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach Ollama Cloud. {exc.reason}") from exc
    message = payload.get("message") or {}
    text = str(message.get("content") or "").strip()
    if not text:
        raise RuntimeError("Ollama Cloud returned an empty reply.")
    return text
