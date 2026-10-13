"""Ollama local and cloud (https://ollama.com).

Supports a local Ollama daemon (`http://127.0.0.1:11434` or `FOURD_MIND_OLLAMA_URL`),
any OpenAI-compatible local server (`FOURD_MIND_CODER_URL`), or Ollama Cloud when
an API key is set (`OLLAMA_API_KEY` or the coder panel).
"""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

CLOUD_API = "https://ollama.com/api"
LOCAL_OLLAMA = "http://127.0.0.1:11434"

# Used when neither local Ollama nor ollama.com can be reached.
FALLBACK_MODELS = (
    "deepseek-coder-v2:16b",
    "deepseek-r1:14b",
    "qwen2.5-coder:7b",
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
    return "code" in lowered or lowered.startswith("gpt-oss") or lowered.startswith("deepseek") or "qwen2.5-coder" in lowered


def order_models(names: list[str]) -> list[dict]:
    rows = [{"name": name, "coder": is_coder(name)} for name in dict.fromkeys(names) if name]
    rows.sort(key=lambda row: (not row["coder"], row["name"]))
    return rows


def default_model(models: list[dict]) -> str | None:
    for row in models:
        if "code" in row["name"].lower() or "deepseek" in row["name"].lower():
            return row["name"]
    return models[0]["name"] if models else None


def local_ollama_url(explicit: str | None = None) -> str:
    url = (explicit or os.environ.get("FOURD_MIND_OLLAMA_URL") or os.environ.get("OLLAMA_HOST") or LOCAL_OLLAMA).strip()
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    return url.rstrip("/")


def list_local_models(base_url: str | None = None, opener=urlopen, timeout: float = 2.0) -> list[dict]:
    """Models installed on the local Ollama daemon (`ollama list`)."""
    base = local_ollama_url(base_url)
    api = base if base.endswith("/api") else f"{base}/api"
    request = Request(f"{api}/tags", headers={"Accept": "application/json"})
    with opener(request, timeout=timeout) as response:
        payload = json.load(response)
    names = []
    for item in payload.get("models") or []:
        name = item.get("name") or item.get("model")
        if name:
            names.append(f"local:{name}")
    return order_models(names)


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


def local_chat(
    model: str,
    messages: list[dict],
    base_url: str | None = None,
    opener=urlopen,
    timeout: float = 180,
) -> str:
    """Chat completion against a local Ollama daemon or OpenAI-compatible server."""
    clean_model = model[len("local:") :] if model.startswith("local:") else model
    base = local_ollama_url(base_url)
    if base.endswith("/v1") or base.endswith("/chat/completions"):
        endpoint = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
        body = json.dumps({"model": clean_model, "messages": messages, "temperature": 0.2}).encode("utf-8")
        request = Request(endpoint, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with opener(request, timeout=timeout) as response:
                payload = json.load(response)
        except URLError as exc:
            raise RuntimeError(f"Could not reach local server at {endpoint}. {exc.reason}") from exc
        return str(payload["choices"][0]["message"]["content"]).strip()
    api = base if base.endswith("/api") else f"{base}/api"
    body = json.dumps({"model": clean_model, "messages": messages, "stream": False}).encode("utf-8")
    request = Request(f"{api}/chat", data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with opener(request, timeout=timeout) as response:
            payload = json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise RuntimeError(f"Local Ollama returned {exc.code}. {detail}".strip()) from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach local Ollama at {base}. {exc.reason}") from exc
    message = payload.get("message") or {}
    text = str(message.get("content") or "").strip()
    if not text:
        raise RuntimeError("Local Ollama returned an empty reply.")
    return text


def chat(model: str, messages: list[dict], key: str, opener=urlopen, timeout: float = 180) -> str:
    """One non-streaming chat completion (local Ollama when `model` starts with `local:`, or Ollama Cloud)."""
    if model.startswith("local:"):
        return local_chat(model, messages, opener=opener, timeout=timeout)
    if not key:
        raise RuntimeError(
            "Set an Ollama Cloud API key, or pick a local model (local:deepseek-coder-v2, local:qwen2.5-coder, or the 4D solid coder)."
        )
    if not model:
        raise RuntimeError("Choose a model.")
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
