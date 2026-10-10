"""A coder that reads the files behind the solid and asks Ollama Cloud for a fix.

The picture is the shape of the code. This module opens the checkout those
dots were rasterized from and sends the relevant source with the question.
"""

from __future__ import annotations

from pathlib import Path

from tesseract.ollama_cloud import chat as cloud_chat

SYSTEM = (
    "You are the coder attached to a 4D view of a git repository. "
    "The picture is not the source text. Each dot is one line of code, placed by file across x, "
    "line position up y, indentation along z, and commit time along t. "
    "You are given the real source of the files that geometry marks as most in need of a fix. "
    "Explain the likely bug and propose a concrete patch, naming the file and the replacement. "
    "Stay inside the excerpts you were shown. "
    "Fix defects. Do not write exploits, payloads, or attack procedures. "
    "If no checkout was provided, say the solid has no source to edit."
)

MAX_FILES = 4
MAX_LINES = 160


class CoderError(RuntimeError):
    pass


def _inside(root: Path, relative: str) -> Path | None:
    root = root.resolve()
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        return None
    if not candidate.is_file():
        return None
    return candidate


def excerpt(root: Path, relative: str, start: int | None = None, end: int | None = None) -> str | None:
    """A bounded slice of a file that stays inside `root`."""
    path = _inside(root, relative)
    if path is None:
        return None
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    if not lines:
        return None
    if start is None or end is None:
        begin, stop = 1, min(len(lines), MAX_LINES)
    else:
        center = max(1, (int(start) + int(end)) // 2)
        begin = max(1, center - MAX_LINES // 2)
        stop = min(len(lines), begin + MAX_LINES - 1)
        begin = max(1, stop - MAX_LINES + 1)
    body = "\n".join(f"{number:4d} | {lines[number - 1]}" for number in range(begin, stop + 1))
    return f"## {relative}  lines {begin}-{stop} of {len(lines)}\n{body}"


def source_context(root: Path | None, meta: dict, hotspots: list[dict], focus: str | None) -> tuple[str, list[str]]:
    """Return `(prompt_block, files_included)`."""
    if root is None:
        return "No checkout is held. The solid has no source files on disk.", []
    wanted: list[tuple[str, int | None, int | None]] = []
    if focus:
        wanted.append((focus, None, None))
    for spot in hotspots:
        name = spot.get("file")
        if not name:
            continue
        span = spot.get("lines") or [None, None]
        wanted.append((str(name), span[0], span[1]))
    for name in meta.get("files") or []:
        wanted.append((str(name), 1, MAX_LINES))
    blocks = []
    used = []
    for relative, start, end in wanted:
        if relative in used or len(used) >= MAX_FILES:
            continue
        block = excerpt(root, relative, start, end)
        if block:
            blocks.append(block)
            used.append(relative)
    if not blocks:
        return "The checkout is held, but none of the mapped files could be read.", []
    return "\n\n".join(blocks), used


def ask(
    meta: dict,
    root: Path | None,
    question: str,
    model: str,
    key: str,
    hotspots: list[dict] | None = None,
    focus: str | None = None,
    chat=cloud_chat,
) -> dict:
    """Ask the cloud coder about the source behind this solid."""
    question = (question or "").strip() or "What is the most likely bug in this code, and what is the fix?"
    if root is None or meta.get("kind") != "repo":
        raise CoderError(
            "This picture has no source to edit. Hold a folder on this computer. "
            "The dots are lines of code placed in space; the coder reads the files those dots stand for."
        )
    if not key:
        raise CoderError("Set an Ollama Cloud API key. Create one under Settings → API keys on ollama.com.")
    block, files = source_context(root, meta, list(hotspots or []), focus)
    reading = meta.get("reading") or ""
    user = (
        f"Repository: {meta.get('name')}\n"
        f"{reading}\n\n"
        f"Source behind the solid:\n{block}\n\n"
        f"Question: {question}"
    )
    text = chat(model, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], key)
    return {"text": text, "model": model, "files": files}
