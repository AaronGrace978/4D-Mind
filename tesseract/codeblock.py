"""A git repository's history, rasterized as one 4D block.

The axes and channels match a life, so the viewer and the block net read it unchanged.

Axes of the returned array are `(t, z, y, x, channel)`.

    t  commit, oldest to newest along first-parent history
    z  nesting depth of the line (its leading indentation)
    y  position of the line in its file, top of the file at the top of the block
    x  file, sorted by path so a directory is a neighborhood

Channels
    0 code: a non-blank line exists here
    1 definitions: def, class, fn, function, struct, ...
    2 churn: the share of lines here that changed since the previous slice
    3 fix lines: written by a commit whose message says fix, bug, or revert, kept while they survive
    4 other authors: lines written by someone other than the main author
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tesseract.life import AFFECT, INVARIANT_NAMES, N_CHANNELS, OCC, OTHER, SCAR, TISSUE, invariants_from_volume

CODE_SUFFIXES = frozenset(
    {
        ".py", ".pyi", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".rs", ".go", ".java", ".kt", ".kts",
        ".scala", ".swift", ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".cs", ".rb", ".php", ".lua",
        ".sh", ".bash", ".zsh", ".css", ".scss", ".html", ".vue", ".svelte", ".sql", ".r", ".jl",
        ".dart", ".ex", ".exs", ".erl", ".hs", ".ml", ".clj", ".zig", ".nim", ".toml", ".yaml", ".yml",
    }
)
FIX_PATTERN = re.compile(r"\b(fix(es|ed)?|bug|hotfix|revert(s|ed)?)\b", re.IGNORECASE)
DEF_PATTERN = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?"
    r"(?:def|class|fn|function|struct|enum|impl|trait|interface|func|module)\b"
)
LABELS = {
    "occupancy": "code",
    "tissue": "definitions",
    "affect": "churn",
    "scar": "fix lines",
    "other": "other authors",
}
MAX_BLOB_BYTES = 1_000_000
# Indentation at or past this many columns lands on the top z layer.
MAX_INDENT = 24


@dataclass
class Commit:
    sha: str
    author: str
    time: int
    subject: str
    merge: bool = False

    @property
    def fix(self) -> bool:
        return bool(FIX_PATTERN.search(self.subject))


@dataclass
class Line:
    text: str
    born: int
    author: str
    fix: bool


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout


class _BlobReader:
    """One long-lived `git cat-file --batch`, so each file version is not a new process."""

    def __init__(self, repo: Path) -> None:
        self.proc = subprocess.Popen(
            ["git", "-C", str(repo), "cat-file", "--batch"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )

    def lines(self, sha: str, path: str) -> list[str] | None:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(f"{sha}:{path}\n".encode("utf-8"))
        self.proc.stdin.flush()
        header = self.proc.stdout.readline().decode("utf-8", "replace").split()
        if len(header) < 3 or header[1] != "blob":
            return None
        size = int(header[2])
        data = self.proc.stdout.read(size)
        self.proc.stdout.read(1)
        if size > MAX_BLOB_BYTES or b"\0" in data:
            return None
        return data.decode("utf-8", "replace").splitlines()

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait(timeout=10)


def _is_code(path: str, suffixes: frozenset[str]) -> bool:
    return Path(path).suffix.lower() in suffixes


def list_commits(repo: Path, ref: str = "HEAD", max_commits: int | None = None) -> list[Commit]:
    raw = _git(repo, "log", "--first-parent", "--reverse", "--format=%H%x1f%an%x1f%at%x1f%P%x1f%s", ref)
    commits = []
    for row in raw.splitlines():
        parts = row.split("\x1f")
        if len(parts) == 5:
            merge = len(parts[3].split()) > 1
            commits.append(Commit(parts[0], parts[1], int(parts[2]), parts[4], merge))
    if max_commits is not None and len(commits) > max_commits:
        commits = commits[-max_commits:]
    return commits


def _changes(repo: Path, previous: str | None, current: str) -> list[tuple[str, str | None, str | None]]:
    """`(status, old_path, new_path)` for each file that differs between two commits."""
    if previous is None:
        names = _git(repo, "ls-tree", "-r", "--name-only", "-z", current).split("\0")
        return [("A", None, name) for name in names if name]
    fields = _git(repo, "diff-tree", "-r", "-M", "--name-status", "-z", previous, current).split("\0")
    out = []
    index = 0
    while index < len(fields) and fields[index]:
        status = fields[index][0]
        if status in "RC":
            out.append((status, fields[index + 1], fields[index + 2]))
            index += 3
        else:
            path = fields[index + 1]
            out.append((status, path if status != "A" else None, path if status != "D" else None))
            index += 2
    return out


def _blame(repo: Path, sha: str, path: str) -> list[tuple[str, bool]] | None:
    """`(author, written_by_a_fix)` for every line of `path` at `sha`."""
    try:
        raw = _git(repo, "blame", "--porcelain", sha, "--", path)
    except subprocess.CalledProcessError:
        return None
    info: dict[str, dict[str, str]] = {}
    out: list[tuple[str, bool]] = []
    current = None
    for row in raw.splitlines():
        if row.startswith("\t"):
            entry = info.get(current or "", {})
            out.append((entry.get("author", ""), bool(FIX_PATTERN.search(entry.get("summary", "")))))
            continue
        head = row.split(" ", 1)[0]
        if len(head) == 40 and all(char in "0123456789abcdef" for char in head):
            current = head
            info.setdefault(current, {})
        elif current and row.startswith("author "):
            info[current]["author"] = row[len("author "):]
        elif current and row.startswith("summary "):
            info[current]["summary"] = row[len("summary "):]
    return out


def _carry(
    old: list[Line],
    new_text: list[str],
    born: int,
    commit: Commit,
    blame: list[tuple[str, bool]] | None = None,
) -> list[Line]:
    """Keep the provenance of lines that survive the edit; new or rewritten lines are born at `born`.

    A merge brings in lines written on another branch, so their author and fix flag come from `blame`.
    """
    matcher = difflib.SequenceMatcher(None, [line.text for line in old], new_text, autojunk=False)
    out: list[Line] = []
    for tag, i0, i1, j0, j1 in matcher.get_opcodes():
        if tag == "equal":
            out.extend(old[i0:i1])
            continue
        for j in range(j0, j1):
            if blame is not None and j < len(blame) and blame[j][0]:
                author, fix = blame[j]
            else:
                author, fix = commit.author, commit.fix
            out.append(Line(new_text[j], born, author, fix))
    return out


def _indent_columns(text: str) -> int:
    columns = 0
    for char in text:
        if char == " ":
            columns += 1
        elif char == "\t":
            columns += 4
        else:
            break
    return columns


def _bins(n_commits: int, n_bins: int) -> list[int]:
    """The commit each time slice shows: the last commit inside that slice's span."""
    return [max(0, min(n_commits - 1, int(round((b + 1) * n_commits / n_bins)) - 1)) for b in range(n_bins)]


def rasterize_repo(
    repo: str | Path,
    shape: tuple[int, int, int, int],
    *,
    ref: str = "HEAD",
    max_commits: int | None = 600,
    suffixes: frozenset[str] = CODE_SUFFIXES,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return `(volume, invariants, meta)` for a repository's history."""
    repo = Path(repo).resolve()
    top = Path(_git(repo, "rev-parse", "--show-toplevel").strip())
    commits = list_commits(top, ref, max_commits)
    if not commits:
        raise ValueError(f"{top} has no commits on {ref}")
    t_size, depth, height, width = shape

    changes = []
    paths: set[str] = set()
    for index, commit in enumerate(commits):
        diff = _changes(top, commits[index - 1].sha if index else None, commit.sha)
        diff = [item for item in diff if any(p and _is_code(p, suffixes) for p in item[1:])]
        changes.append(diff)
        paths.update(p for item in diff for p in item[1:] if p and _is_code(p, suffixes))
    files = sorted(paths)
    column = {path: k for k, path in enumerate(files)}
    bins = _bins(len(commits), t_size)

    reader = _BlobReader(top)
    state: dict[str, list[Line]] = {}
    snapshots: dict[int, dict[str, list[Line]]] = {}
    max_lines = 1
    try:
        for index, commit in enumerate(commits):
            for status, old_path, new_path in changes[index]:
                previous = state.pop(old_path, []) if old_path else []
                if status == "C" and old_path:
                    state[old_path] = previous
                if new_path is None or not _is_code(new_path, suffixes):
                    continue
                text = reader.lines(commit.sha, new_path)
                if text is None:
                    continue
                blame = _blame(top, commit.sha, new_path) if commit.merge else None
                state[new_path] = _carry(previous, text, index, commit, blame)
                max_lines = max(max_lines, len(text))
            if index in bins:
                snapshots[index] = {path: list(lines) for path, lines in state.items()}
    finally:
        reader.close()

    written = Counter(line.author for lines in state.values() for line in lines if line.text.strip())
    main_author = written.most_common(1)[0][0] if written else commits[-1].author
    volume = np.zeros((t_size, depth, height, width, N_CHANNELS), dtype=np.float32)
    bands = _file_bands(len(files), width)
    for b, commit_index in enumerate(bins):
        since = bins[b - 1] if b else -1
        _paint_frame(volume[b], snapshots[commit_index], column, bands, max_lines, since, main_author)

    invariants = invariants_from_volume(volume)
    meta = _meta(top, commits, bins, files, bands, max_lines, main_author, sorted(written), volume, invariants)
    return volume, invariants, meta


def _file_bands(n_files: int, width: int) -> list[tuple[int, int]]:
    bands = []
    for k in range(max(n_files, 1)):
        x0 = int(k * width / max(n_files, 1))
        x1 = max(x0 + 1, int((k + 1) * width / max(n_files, 1)))
        if x1 - x0 >= 3:
            x1 -= 1
        bands.append((x0, min(x1, width)))
    return bands


def _paint_frame(
    frame: np.ndarray,
    files: dict[str, list[Line]],
    column: dict[str, int],
    bands: list[tuple[int, int]],
    max_lines: int,
    since: int,
    main_author: str,
) -> None:
    depth, height, _width, _channels = frame.shape
    count = np.zeros((depth, height), dtype=np.float32)
    for path, lines in files.items():
        if path not in column or not lines:
            continue
        x0, x1 = bands[column[path]]
        count[:] = 0.0
        churn = np.zeros_like(count)
        plane = np.zeros((depth, height, N_CHANNELS), dtype=np.float32)
        for number, line in enumerate(lines):
            if not line.text.strip():
                continue
            y = height - 1 - min(int(number * height / max_lines), height - 1)
            z = min(int(round(min(_indent_columns(line.text), MAX_INDENT) / MAX_INDENT * (depth - 1))), depth - 1)
            count[z, y] += 1.0
            churn[z, y] += 1.0 if line.born > since else 0.0
            plane[z, y, OCC] = 1.0
            if DEF_PATTERN.match(line.text):
                plane[z, y, TISSUE] = 1.0
            if line.fix:
                plane[z, y, SCAR] = 1.0
            if line.author != main_author:
                plane[z, y, OTHER] = 1.0
        plane[..., AFFECT] = np.divide(churn, count, out=np.zeros_like(churn), where=count > 0)
        frame[:, :, x0:x1, :] = np.maximum(frame[:, :, x0:x1, :], plane[:, :, None, :])


def _repo_name(top: Path) -> str:
    try:
        url = _git(top, "remote", "get-url", "origin").strip()
    except subprocess.CalledProcessError:
        return top.name
    name = url.rstrip("/").rsplit("/", 1)[-1].rsplit(":", 1)[-1]
    return name[:-4] if name.endswith(".git") else (name or top.name)


def _iso(stamp: int) -> str:
    return datetime.fromtimestamp(stamp, tz=timezone.utc).strftime("%Y-%m-%d")


def _meta(
    top: Path,
    commits: list[Commit],
    bins: list[int],
    files: list[str],
    bands: list[tuple[int, int]],
    max_lines: int,
    main_author: str,
    authors: list[str],
    volume: np.ndarray,
    invariants: np.ndarray,
) -> dict:
    slices = [
        {
            "sha": commits[i].sha[:7],
            "subject": commits[i].subject,
            "author": commits[i].author,
            "date": _iso(commits[i].time),
            "fix": commits[i].fix,
        }
        for i in bins
    ]
    churn = volume[..., AFFECT].reshape(volume.shape[0], -1).sum(axis=1)
    meta = {
        "kind": "repo",
        "name": _repo_name(top),
        "other_name": "other authors",
        "main_author": main_author,
        "authors": authors,
        "same_room": False,
        "n_commits": len(commits),
        "first_date": _iso(commits[0].time),
        "last_date": _iso(commits[-1].time),
        "files": files,
        "bands": [list(band) for band in bands[: len(files)]],
        "slices": slices,
        "max_lines": max_lines,
        "max_indent": MAX_INDENT,
        "churn_peak_bin": int(churn[1:].argmax()) + 1 if len(churn) > 1 else 0,
        "invariants": invariants.tolist(),
        "invariant_names": list(INVARIANT_NAMES),
        "labels": dict(LABELS),
    }
    meta["reading"] = repo_reading(meta)
    return meta


def _slice_at(meta: dict, t: float) -> dict:
    slices = meta["slices"]
    return slices[min(len(slices) - 1, max(0, int(round(float(t) * (len(slices) - 1)))))]


def _commit_text(entry: dict) -> str:
    return f"{entry['sha']} \u201c{entry['subject']}\u201d ({entry['date']})"


def repo_reading(meta: dict) -> str:
    inv = meta["invariants"]
    parts = [
        f"{meta['name']} as one solid: {meta['n_commits']} commits from {meta['first_date']} to "
        f"{meta['last_date']} along t, {len(meta['files'])} files across x, line position up y, "
        "nesting depth along z.",
    ]
    if inv[1] >= 0:
        parts.append(
            f"The first fix opens at t={inv[1]:.2f}, {_commit_text(_slice_at(meta, inv[1]))}. "
            "Its lines stay in the solid until someone deletes them."
        )
    else:
        parts.append("No commit message in this history says fix, bug, or revert.")
    if inv[2] >= 0:
        parts.append(
            f"Code by other authors first appears at t={inv[2]:.2f}, {_commit_text(_slice_at(meta, inv[2]))}."
        )
    else:
        parts.append(f"Every surviving line was written by {meta['main_author']}.")
    if len(meta["slices"]) > 1:
        peak = meta["churn_peak_bin"]
        t_peak = peak / (len(meta["slices"]) - 1)
        parts.append(f"Churn peaks at t={t_peak:.2f}, {_commit_text(meta['slices'][peak])}.")
    return "\n\n".join(parts)


def repo_probe_sentence(meta: dict, channels: dict[str, float], x: float, y: float, z: float, t: float) -> str:
    entry = _slice_at(meta, t)
    bits = [f"t={t:.2f} is {_commit_text(entry)}."]
    files = meta["files"]
    if files:
        width = max(band[1] for band in meta["bands"])
        column = min(int(float(x) * width), width - 1)
        name = next((files[k] for k, band in enumerate(meta["bands"]) if band[0] <= column < band[1]), None)
        line = int(round((1.0 - float(y)) * meta["max_lines"])) + 1
        indent = int(round(float(z) * meta["max_indent"]))
        if name:
            bits.append(f"x={x:.2f} is {name}, around line {line}, indented about {indent} columns.")
    if channels["occupancy"] > 0.3:
        bits.append("Code exists here.")
    else:
        bits.append("No code is here at this commit.")
    if channels["tissue"] > 0.3:
        bits.append("A definition sits here.")
    if channels["scar"] > 0.3:
        bits.append("Lines written by a fix pass through this point.")
    if channels["other"] > 0.3:
        bits.append("Another author wrote lines here.")
    bits.append(f"Changed since the previous slice: {channels['affect']:.0%}.")
    return " ".join(bits)


def export_repo_npz(path: str | Path, repo: str | Path, shape: tuple[int, int, int, int], **kwargs) -> dict:
    volume, invariants, meta = rasterize_repo(repo, shape, **kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, volume=volume, invariants=invariants, meta=np.asarray(json.dumps(meta)))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Rasterize a git repository's history into a 4D block")
    parser.add_argument("repo", nargs="?", default=".")
    parser.add_argument("--out", default="repo_block.npz")
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--max-commits", type=int, default=600)
    parser.add_argument("--shape", type=int, nargs=4, default=[16, 32, 32, 32], metavar=("T", "D", "H", "W"))
    args = parser.parse_args()
    meta = export_repo_npz(args.out, args.repo, tuple(args.shape), ref=args.ref, max_commits=args.max_commits)
    print(meta["reading"])
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
