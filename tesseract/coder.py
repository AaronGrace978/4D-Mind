"""A 4D-Lens coder that reads the files behind the solid and reasons across spacetime.

The picture is the shape of the code. Each dot is one line of code, placed by
file across x, line position up y, indentation along z, and commit time along t.
This module computes a 4D spatiotemporal lens over the volume (coupled churn across
files, persistent fix filaments, nesting ridges, and causal vs eternalist signals),
opens the checkout those dots were rasterized from, and answers with:
  1. `grounded:4d-lens` — deterministic 4D geometric code audit (needs no API key)
  2. `lora:4d-coder` — local GPU LoRA model conditioned on the 4D block soft prompts
  3. `local:<model>` — local Ollama daemon (e.g. DeepSeek-Coder-V2, DeepSeek-R1, Qwen2.5-Coder)
  4. Ollama Cloud model — when an API key is provided
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

from tesseract.life import AFFECT, OCC, OTHER, SCAR, TISSUE
from tesseract.ollama_cloud import chat as cloud_chat

SYSTEM = (
    "You are a 4D-Lens Coder attached to a spacetime solid of a git repository. "
    "The repository is not a flat set of files: it is a 4D volume (t=commit history, "
    "x=file neighborhood, y=line position, z=indentation depth). "
    "You see the 4D geometric lens (where fix filaments open and persist along t, which "
    "files co-mutate in the same time slices, where indentation ridges z collide with "
    "high churn, and where the causal witness vs eternalist resident diverge) alongside "
    "the real source of the files that geometry marks as most in need of a fix. "
    "Reason through this 4D lens first: explain how the bug's spacetime trajectory "
    "(churn + scar filaments + cross-file coupling) points to the defect, then propose a "
    "concrete patch naming the file and the replacement. Stay inside the excerpts you were shown. "
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


def fourd_code_lens(
    volume: np.ndarray | None,
    meta: dict,
    hotspots: list[dict] | None = None,
    focus: str | None = None,
) -> dict:
    """Compute a 4D spatiotemporal lens over a repository block.

    Extracts:
      - worldline trajectories per file (churn along t, persistent fix-scar density, z-nesting peak)
      - cross-file co-churn coupling (files that mutate in the same time slices along t)
      - temporal fracture slices (commit bins where churn and fix scars spike together)
      - a structured 4D reading suitable for prompts or standalone grounded inspection
    """
    files = list(meta.get("files") or [])
    bands = list(meta.get("bands") or [])
    slices = list(meta.get("slices") or [])
    spots = list(hotspots or [])
    if volume is None or volume.ndim != 5 or not files or not bands:
        summary = meta.get("reading") or "4D Spatiotemporal Code Lens: no spacetime volume attached."
        return {
            "summary": summary,
            "n_commits": int(meta.get("n_commits", 0)),
            "worldlines": [],
            "file_worldlines": [],
            "coupled_pairs": [],
            "fracture_slices": [],
            "focus": focus,
        }

    t_size, depth, _height, _width, _ch = volume.shape
    z_grid = np.linspace(0.0, float(meta.get("max_indent", 24)), depth, dtype=np.float32)
    worldlines: list[dict] = []
    churn_series: list[np.ndarray] = []

    for name, band in zip(files, bands):
        x0, x1 = int(band[0]), int(band[1])
        slab = volume[:, :, :, x0:x1, :]
        occ = slab[..., OCC] > 0.3
        if not occ.any():
            churn_series.append(np.zeros(t_size, dtype=np.float32))
            continue
        t_churn = slab[..., AFFECT].reshape(t_size, -1).mean(axis=1)
        churn_series.append(t_churn)
        scar_share = float((slab[-1, ..., SCAR] > 0.3).sum() / max((slab[-1, ..., OCC] > 0.3).sum(), 1))
        other_share = float((slab[-1, ..., OTHER] > 0.3).sum() / max((slab[-1, ..., OCC] > 0.3).sum(), 1))
        defs_share = float((slab[-1, ..., TISSUE] > 0.3).sum() / max((slab[-1, ..., OCC] > 0.3).sum(), 1))
        z_mass = slab[-1, ..., OCC].sum(axis=(1, 2))
        mean_indent = float((z_mass * z_grid).sum() / max(z_mass.sum(), 1e-6))
        first_t = int(np.argmax(occ.reshape(t_size, -1).any(axis=1)))
        peak_t = int(np.argmax(t_churn))
        worldlines.append(
            {
                "file": name,
                "born_t": round(first_t / max(t_size - 1, 1), 2),
                "peak_churn_t": round(peak_t / max(t_size - 1, 1), 2),
                "mean_churn": round(float(t_churn.mean()), 3),
                "scar_density": round(scar_share, 3),
                "other_author_density": round(other_share, 3),
                "def_density": round(defs_share, 3),
                "mean_indent_cols": round(mean_indent, 1),
            }
        )

    coupled_pairs: list[dict] = []
    if len(churn_series) >= 2 and t_size >= 2:
        mat = np.stack(churn_series, axis=0)
        active = mat > 0.02
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                both = int((active[i] & active[j]).sum())
                either = int((active[i] | active[j]).sum())
                if both > 0 and either > 0:
                    jaccard = both / either
                    coupled_pairs.append(
                        {
                            "files": [files[i], files[j]],
                            "coupling": round(float(jaccard), 2),
                            "shared_slices": both,
                        }
                    )
        coupled_pairs.sort(key=lambda item: (-item["coupling"], -item["shared_slices"]))

    fracture_slices: list[dict] = []
    slice_churn = volume[..., AFFECT].reshape(t_size, -1).mean(axis=1)
    slice_scar = (volume[..., SCAR] > 0.3).reshape(t_size, -1).mean(axis=1)
    order = np.argsort(-(slice_churn + slice_scar))[: min(3, t_size)]
    for idx in order:
        b = int(idx)
        t_val = round(b / max(t_size - 1, 1), 2)
        commit_info = slices[min(b, len(slices) - 1)] if slices else {}
        fracture_slices.append(
            {
                "t": t_val,
                "sha": commit_info.get("sha", ""),
                "subject": commit_info.get("subject", ""),
                "churn": round(float(slice_churn[b]), 3),
                "scar": round(float(slice_scar[b]), 3),
            }
        )

    worldlines.sort(key=lambda item: -(item["scar_density"] * 0.5 + item["mean_churn"] * 0.5))
    lines = ["4D Spatiotemporal Code Lens (t=commit, x=file, y=line, z=indent):"]
    for wl in worldlines[:4]:
        lines.append(
            f"- {wl['file']}: born t={wl['born_t']:.2f}, peak churn t={wl['peak_churn_t']:.2f}, "
            f"fix-scar density {wl['scar_density']:.0%}, mean indent {wl['mean_indent_cols']:.1f} cols, "
            f"other-author share {wl['other_author_density']:.0%}"
        )
    if coupled_pairs:
        top_c = coupled_pairs[:3]
        pairs_str = ", ".join(f"{p['files'][0]} <-> {p['files'][1]} ({p['coupling']:.0%} co-churn)" for p in top_c)
        lines.append(f"- Cross-file 4D coupling along t: {pairs_str}")
    if fracture_slices:
        frac_str = "; ".join(
            f"t={f['t']:.2f} {f['sha']} \"{f['subject']}\" (churn {f['churn']:.3f}, scar {f['scar']:.1%})"
            if f["sha"]
            else f"t={f['t']:.2f} (churn {f['churn']:.3f}, scar {f['scar']:.1%})"
            for f in fracture_slices[:2]
        )
        lines.append(f"- Temporal fracture slices: {frac_str}")
    if spots:
        spot_str = ", ".join(
            f"{s['file']}:{s['lines'][0]}-{s['lines'][1]} (risk {s.get('score', 0.0):.2f})"
            if s.get("lines")
            else f"{s['file']} (risk {s.get('score', 0.0):.2f})"
            for s in spots[:4]
        )
        lines.append(f"- Causal future-fix hotspots at newest slice: {spot_str}")

    return {
        "summary": "\n".join(lines),
        "n_commits": int(meta.get("n_commits", t_size)),
        "worldlines": worldlines,
        "file_worldlines": worldlines,
        "coupled_pairs": coupled_pairs[:5],
        "fracture_slices": fracture_slices,
        "focus": focus,
    }


def ablation_lens(volume: np.ndarray | None, meta: dict, cut_t: float = 0.5) -> dict:
    """Compare the Eternalist 4D block view (full t) against a Causal 3D witness truncated at `cut_t`.

    Measures what the non-causal 4D block knows about the repository's future evolution
    (future fix-scars that open after `cut_t`, late-stage co-churn coupling, and structural
    worldline survival) that a causal witness standing at `cut_t` cannot see.
    """
    files = list(meta.get("files") or [])
    bands = list(meta.get("bands") or [])
    if volume is None or volume.ndim != 5 or not files or not bands:
        return {
            "cut_t": round(float(cut_t), 2),
            "blind_future_scars": [],
            "hidden_couplings": [],
            "ablation_gain": 0.0,
            "summary": "4D Ablation Lens: hold a repository or sculpture solid to compare Eternalist vs Causal views.",
        }
    t_size = volume.shape[0]
    cut_bin = max(1, min(t_size - 1, int(round(float(cut_t) * (t_size - 1)))))
    causal_vol = volume[:cut_bin]
    future_vol = volume[cut_bin:]

    blind_future_scars: list[dict] = []
    for name, band in zip(files, bands):
        x0, x1 = int(band[0]), int(band[1])
        early_scar = float((causal_vol[:, :, :, x0:x1, SCAR] > 0.3).mean())
        late_scar = float((future_vol[:, :, :, x0:x1, SCAR] > 0.3).mean())
        late_churn = float(future_vol[:, :, :, x0:x1, AFFECT].mean())
        delta = late_scar - early_scar
        if delta > 0.005 or late_scar > 0.02:
            blind_future_scars.append(
                {
                    "file": name,
                    "causal_scar": round(early_scar, 4),
                    "eternalist_future_scar": round(late_scar, 4),
                    "future_churn": round(late_churn, 4),
                    "blind_delta": round(delta, 4),
                }
            )
    blind_future_scars.sort(key=lambda item: (-item["blind_delta"], -item["eternalist_future_scar"]))

    full_lens = fourd_code_lens(volume, meta)
    causal_lens = fourd_code_lens(causal_vol, meta)
    causal_pair_set = {tuple(p["files"]) for p in causal_lens.get("coupled_pairs", [])}
    hidden_couplings = [
        p for p in full_lens.get("coupled_pairs", []) if tuple(p["files"]) not in causal_pair_set
    ]
    total_early = float((causal_vol[..., SCAR] > 0.3).mean())
    total_full = float((volume[..., SCAR] > 0.3).mean())
    gain = round(max(0.0, total_full - total_early) / max(total_full, 1e-6), 3)

    lines = [
        f"4D Ablation Lens (Eternalist Full Block vs Causal Witness stopped at t={cut_t:.2f}, slice {cut_bin}/{t_size}):",
        f"- Non-causal 4D foresight gain over causal witness: {gain:.0%} of fix-scar structure lies beyond the causal horizon.",
    ]
    if blind_future_scars:
        top_b = ", ".join(
            f"{b['file']} (causal scar {b['causal_scar']:.1%} -> future scar {b['eternalist_future_scar']:.1%})"
            for b in blind_future_scars[:3]
        )
        lines.append(f"- Future fix-scars invisible to causal witness at t={cut_t:.2f}: {top_b}")
    if hidden_couplings:
        top_h = ", ".join(f"{p['files'][0]} <-> {p['files'][1]} ({p['coupling']:.0%})" for p in hidden_couplings[:2])
        lines.append(f"- Emergent cross-file couplings after t={cut_t:.2f}: {top_h}")

    return {
        "cut_t": round(float(cut_t), 2),
        "cut_bin": cut_bin,
        "t_size": t_size,
        "blind_future_scars": blind_future_scars[:5],
        "hidden_couplings": hidden_couplings[:4],
        "ablation_gain": gain,
        "summary": "\n".join(lines),
    }


def topology_metrics(volume: np.ndarray) -> dict:
    """Compact 4D topological health metrics for before/after refactor verification."""
    from tesseract.risk import geometric_risk

    field = geometric_risk(volume)
    latest_occ = volume[-1, ..., OCC] > 0.3
    latest_risk = float(np.where(latest_occ, field[-1], 0.0).mean())
    max_risk = float(np.where(latest_occ, field[-1], 0.0).max()) if latest_occ.any() else 0.0
    z_depth = volume.shape[1]
    z_weights = np.linspace(0.0, 1.0, z_depth, dtype=np.float32)[:, None, None]
    indent_load = float((volume[-1, ..., OCC] * z_weights).sum() / max(volume[-1, ..., OCC].sum(), 1e-6))
    scar_load = float((volume[-1, ..., SCAR] > 0.3).mean())
    return {
        "mean_risk": round(latest_risk, 4),
        "peak_risk": round(max_risk, 4),
        "indent_load": round(indent_load, 4),
        "scar_load": round(scar_load, 4),
    }


def grounded_code_review(
    meta: dict,
    root: Path | None,
    question: str,
    lens: dict | None = None,
    hotspots: list[dict] | None = None,
    focus: str | None = None,
) -> tuple[str, list[str]]:
    """Deterministic 4D geometric code analysis and patch guide that runs locally without any API key."""
    spots = list(hotspots or [])
    if lens is None:
        lens = fourd_code_lens(None, meta, spots, focus)
    block, files = source_context(root, meta, spots, focus)
    parts = [
        f"4D Spatiotemporal Code Audit for {meta.get('name', 'repository')} (query: {question})",
        "",
        lens["summary"],
    ]
    target_spot = None
    if focus:
        target_spot = next((s for s in spots if s.get("file") == focus), None)
    if target_spot is None and spots:
        target_spot = spots[0]
    if target_spot:
        span = target_spot.get("lines") or [1, 20]
        parts.append(
            f"\nPrimary 4D hotspot: `{target_spot['file']}` lines {span[0]}-{span[1]} "
            f"(causal fix-risk score {float(target_spot.get('score', 0.0)):.2f}). "
            "In the 4D volume, this coordinate combines accumulated churn along t, deep z-indentation, "
            "and surviving fix-scar filaments."
        )
    elif files:
        parts.append(f"\nFocused file in the x-band: `{files[0]}`.")

    if root is not None and files:
        target_file = target_spot["file"] if target_spot and target_spot.get("file") in files else files[0]
        path = _inside(root, target_file)
        if path is not None:
            raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            span = (target_spot.get("lines") if target_spot and target_spot.get("file") == target_file else None) or [
                1,
                min(len(raw_lines), 25),
            ]
            l0 = max(1, min(int(span[0]), len(raw_lines)))
            l1 = max(l0, min(int(span[1]), len(raw_lines)))
            snippet = "\n".join(f"  {ln:4d} | {raw_lines[ln - 1]}" for ln in range(l0, min(l1 + 1, l0 + 16)))
            parts.append(f"\nSource slice at `{target_file}:{l0}-{l1}`:\n{snippet}")
            coupled = [
                p["files"][1] if p["files"][0] == target_file else p["files"][0]
                for p in lens.get("coupled_pairs", [])
                if target_file in p["files"]
            ]
            if coupled:
                parts.append(
                    f"\n4D Refactoring Guidance: `{target_file}` co-mutates along t with {', '.join(coupled[:2])}. "
                    "When patching this region, verify callers/callees across both worldlines so a fix here "
                    "does not open a new scar filament in the coupled file."
                )
    else:
        parts.append(f"\n{block}")
    return "\n".join(parts), files


def ask(
    meta: dict,
    root: Path | None,
    question: str,
    model: str,
    key: str,
    hotspots: list[dict] | None = None,
    focus: str | None = None,
    volume: np.ndarray | None = None,
    voice=None,
    chat=cloud_chat,
) -> dict:
    """Ask the 4D coder about the source behind this solid."""
    question = (question or "").strip() or "What is the most likely bug in this code, and what is the fix?"
    chosen = (model or "").strip() or "grounded:4d-lens"
    spots = list(hotspots or [])
    lens = fourd_code_lens(volume, meta, spots, focus)

    if chosen == "grounded:4d-lens" and meta.get("kind") in {"repo", "sculpture"}:
        text, files = grounded_code_review(meta, root if meta.get("kind") == "repo" else None, question, lens, spots, focus)
        return {"text": text, "reply": text, "model": chosen, "files": files, "lens": lens}

    if root is None or meta.get("kind") != "repo":
        raise CoderError(
            "This picture has no source to edit. Hold a folder on this computer. "
            "The dots are lines of code placed in space; the coder reads the files those dots stand for."
        )

    block, files = source_context(root, meta, spots, focus)
    reading = meta.get("reading") or ""

    if chosen == "lora:4d-coder" and voice is not None and getattr(voice, "name", "") == "lora":
        prompt = (
            f"{lens['summary']}\n\n"
            f"Source behind the solid:\n{block}\n\n"
            f"Question: {question}"
        )
        text = voice.say(meta, 1.0, prompt, [], volume)
        return {"text": text, "reply": text, "model": chosen, "files": files, "lens": lens}

    if not key and not chosen.startswith("local:"):
        raise CoderError(
            "Set an Ollama Cloud API key, or choose 'grounded:4d-lens' / a 'local:' Ollama model (such as local:deepseek-coder-v2) for free local coding."
        )
    user = (
        f"Repository: {meta.get('name')}\n"
        f"{reading}\n\n"
        f"{lens['summary']}\n\n"
        f"Source behind the solid:\n{block}\n\n"
        f"Question: {question}"
    )
    text = chat(chosen, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], key)
    return {"text": text, "reply": text, "model": chosen, "files": files, "lens": lens}


HARNESS_COMMANDS: dict[str, list[str]] = {
    "pytest": [sys.executable, "-m", "pytest", "-q"],
    "git_status": ["git", "status", "-s"],
    "git_diff": ["git", "diff", "--stat"],
    "cargo_check": ["cargo", "check"],
    "npm_test": ["npm", "test"],
}


def run_harness(
    root: Path | None,
    action: str = "pytest",
    file: str | None = None,
    find_text: str | None = None,
    replace_text: str | None = None,
    auto_verify: bool = True,
    rollback_on_fail: bool = False,
    verify_action: str = "pytest",
    timeout: int = 30,
) -> dict:
    """Local workspace harness for building, testing, and self-validating patches inside the held checkout."""
    if root is None or not root.is_dir():
        raise CoderError("Hold a local git repository checkout before running the workspace harness.")
    root = root.resolve()
    action = (action or "").strip().lower()

    if action == "read":
        rel = (file or "").strip()
        if not rel:
            raise CoderError("Specify a relative file path inside the checkout to read.")
        target = _inside(root, rel)
        if target is None:
            raise CoderError(f"Refusing path outside checkout or missing file: {rel}")
        content = target.read_text(encoding="utf-8", errors="replace")
        return {
            "action": "read",
            "file": rel,
            "ok": True,
            "returncode": 0,
            "content": content,
            "output": f"Loaded {rel} ({len(content.splitlines())} lines).",
        }

    if action == "patch":
        rel = (file or "").strip()
        if not rel:
            raise CoderError("Specify a relative file path inside the checkout to patch.")
        target = _inside(root, rel)
        if target is None:
            raise CoderError(f"Refusing path outside checkout or missing file: {rel}")
        if not find_text:
            raise CoderError("Provide the exact snippet (`find_text`) to replace.")
        current = target.read_text(encoding="utf-8")
        if find_text not in current:
            raise CoderError(f"`find_text` was not found in {rel}.")
        updated = current.replace(find_text, replace_text or "", 1)
        target.write_text(updated, encoding="utf-8")
        patch_msg = f"Patched {rel} ({len(find_text)} chars -> {len(replace_text or '')} chars)."
        verification = None
        rolled_back = False
        if auto_verify and verify_action in HARNESS_COMMANDS:
            verification = run_harness(root, action=verify_action, auto_verify=False, timeout=timeout)
            if not verification["ok"] and rollback_on_fail:
                target.write_text(current, encoding="utf-8")
                rolled_back = True
                patch_msg += f"\nVerification ({verify_action}) FAILED — automatically rolled back {rel}."
            elif verification["ok"]:
                patch_msg += f"\nSelf-Validating 4D CI ({verify_action}): PASSED."
            else:
                patch_msg += f"\nSelf-Validating 4D CI ({verify_action}): FAILED (exit {verification['returncode']})."
        return {
            "action": "patch",
            "file": rel,
            "ok": True if verification is None else (verification["ok"] and not rolled_back),
            "patched": not rolled_back,
            "rolled_back": rolled_back,
            "returncode": 0 if verification is None else verification["returncode"],
            "verification": verification,
            "output": patch_msg + (f"\n\n{verification['output']}" if verification else ""),
        }

    if action not in HARNESS_COMMANDS:
        allowed = ", ".join(sorted([*HARNESS_COMMANDS.keys(), "patch", "read"]))
        raise CoderError(f"Unknown harness action {action!r}. Allowed: {allowed}")

    cmd = list(HARNESS_COMMANDS[action])
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        combined = ((proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")).strip()
        if len(combined) > 6000:
            combined = combined[:6000] + "\n... [truncated]"
        return {
            "action": action,
            "command": " ".join(shlex.quote(part) for part in cmd),
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "output": combined or "(completed with no output)",
        }
    except FileNotFoundError as exc:
        raise CoderError(f"Tool not installed on this machine for {action}: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise CoderError(f"Harness command {action} timed out after {timeout}s.") from exc

