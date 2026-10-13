"""Where in a repository's solid will the next fix land?

A fix commit leaves a filament in the scar channel. Looking back over a history,
every place where a new filament was born marks code that needed a fix. The
predictor stands at one time bin, sees only that bin and earlier ones, and ranks
each spatial patch (a band of files, a run of lines, a nesting depth) by how
likely a fix is to land there later.

Two predictors share the targets and the scores:

    heuristic  geometry alone: churn so far, fix filaments nearby, other authors'
               lines, definitions, and nesting depth. Needs only numpy.
    model      the causal risk head of `tesseract.model.Eternalist`, trained on
               real histories with `python -m tesseract.risk train`.

    python -m tesseract.risk scan path/to/repo [--ckpt checkpoints/risk.pt]
    python -m tesseract.risk backtest path/to/repo [--ckpt checkpoints/risk.pt]
    python -m tesseract.risk train repo_a repo_b --gallery doom quake --out checkpoints/risk.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tesseract.life import AFFECT, OCC, OTHER, SCAR, TISSUE
from tesseract.metrics import risk_scores
from tesseract.viewdata import resample_volume

DEFAULT_SHAPE = (16, 8, 32, 32)
WEIGHTS = {"churn": 0.35, "fixes": 0.25, "others": 0.15, "definitions": 0.10, "depth": 0.15}


def new_fix(volume: np.ndarray) -> np.ndarray:
    """`(t, z, y, x)` True where a fix filament appears that the previous slice did not have."""
    scar = volume[..., SCAR] > 0.5
    before = np.concatenate([np.zeros_like(scar[:1]), scar[:-1]], axis=0)
    return scar & ~before


def _token_grid(array: np.ndarray, patch: int) -> np.ndarray:
    """`(t, z, y, x)` -> `(n_t, p, n_z, p, n_y, p, n_x, p)`."""
    time, depth, height, width = array.shape
    return array.reshape(time // patch, patch, depth // patch, patch, height // patch, patch, width // patch, patch)


def risk_targets(volume: np.ndarray, patch: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-token labels and validity, flattened in the model's `(t, z, y, x)` token order.

    A token's label is whether a fix lands in its spatial patch in any slice after its
    time bin. Only tokens that hold code and have a future count.
    """
    time = volume.shape[0]
    n_t = time // patch
    born = new_fix(volume)
    depth, height, width = born.shape[1:]
    spatial = born.reshape(time, depth // patch, patch, height // patch, patch, width // patch, patch).any(axis=(2, 4, 6))
    labels = np.zeros((n_t,) + spatial.shape[1:], dtype=bool)
    later = np.zeros(spatial.shape[1:], dtype=bool)
    for b in reversed(range(n_t)):
        labels[b] = later
        later = later | spatial[b * patch : (b + 1) * patch].any(axis=0)
    valid = _token_grid(volume[..., OCC] > 0.5, patch).any(axis=(1, 3, 5, 7))
    valid[-1] = False
    return labels.reshape(-1), valid.reshape(-1)


def _box(array: np.ndarray, axes: tuple[int, ...] = (1, 2, 3)) -> np.ndarray:
    out = array.astype(np.float32)
    for axis in axes:
        padded = np.pad(out, [(1, 1) if a == axis else (0, 0) for a in range(out.ndim)], mode="edge")
        size = out.shape[axis]
        out = (
            np.take(padded, range(0, size), axis=axis)
            + np.take(padded, range(1, size + 1), axis=axis)
            + np.take(padded, range(2, size + 2), axis=axis)
        ) / 3.0
    return out


def _unit(array: np.ndarray) -> np.ndarray:
    peak = float(array.max()) if array.size else 0.0
    return array / peak if peak > 0 else array


def geometric_risk(volume: np.ndarray) -> np.ndarray:
    """`(t, z, y, x)` risk in `[0, 1]`. Each slice uses only that slice and earlier ones."""
    time, depth = volume.shape[:2]
    code = (volume[..., OCC] > 0.5).astype(np.float32)
    steps = np.arange(1, time + 1, dtype=np.float32).reshape(-1, 1, 1, 1)
    churn = np.cumsum(volume[..., AFFECT], axis=0) / steps
    depth_ramp = (np.arange(depth, dtype=np.float32) / max(depth - 1, 1)).reshape(1, -1, 1, 1)
    parts = {
        "churn": _unit(_box(churn)),
        "fixes": _unit(_box(volume[..., SCAR])),
        "others": _unit(_box(volume[..., OTHER])),
        "definitions": _unit(_box(volume[..., TISSUE])),
        "depth": np.broadcast_to(depth_ramp, code.shape),
    }
    raw = sum(WEIGHTS[name] * part for name, part in parts.items())
    return (_unit(raw * code)).astype(np.float32)


def heuristic_token_scores(volume: np.ndarray, patch: int) -> np.ndarray:
    """The heuristic at the last slice of each token's bin, averaged over that token's code."""
    risk = geometric_risk(volume)
    code = (volume[..., OCC] > 0.5).astype(np.float32)
    ends = np.arange(patch - 1, volume.shape[0], patch)
    at_end_risk = np.repeat(risk[ends], patch, axis=0)
    at_end_code = np.repeat(code[ends], patch, axis=0)
    total = _token_grid(at_end_risk * at_end_code, patch).sum(axis=(1, 3, 5, 7))
    count = _token_grid(at_end_code, patch).sum(axis=(1, 3, 5, 7))
    return np.divide(total, count, out=np.zeros_like(total), where=count > 0).reshape(-1)


def model_risk(model, volume: np.ndarray) -> np.ndarray:
    """`(t, z, y, x)` fix probability from the causal risk head, painted over each token's patch."""
    import torch

    from tesseract.train import volumes_to_torch

    model.eval()
    device = next(model.parameters()).device
    block = volume if tuple(volume.shape[:4]) == model.cfg.shape else resample_volume(volume, model.cfg.shape)
    with torch.no_grad():
        logits = model.risk_logits(volumes_to_torch(block[None].astype(np.float32), device))
    grid = torch.sigmoid(logits[0].float()).cpu().numpy().reshape(model.n_t, model.n_z, model.n_y, model.n_x)
    p = model.cfg.patch
    painted = grid.repeat(p, 0).repeat(p, 1).repeat(p, 2).repeat(p, 3)
    if painted.shape != tuple(volume.shape[:4]):
        painted = resample_volume(painted[..., None], tuple(volume.shape[:4]))[..., 0]
    return painted.astype(np.float32)


def _rows_to_lines(rows: list[int], height: int, max_lines: int) -> tuple[int, int]:
    """1-based line numbers covered by block rows. The top of a file sits at the highest row."""
    first = int((height - 1 - max(rows)) * max_lines / height) + 1
    last = int((height - min(rows)) * max_lines / height)
    return first, max(first, last)


def hotspots(volume: np.ndarray, meta: dict, risk: np.ndarray, top: int = 12) -> dict:
    """Rank files and line ranges at the newest slice by risk."""
    files = meta.get("files", [])
    bands = meta.get("bands", [])
    lengths = meta.get("file_lines") or [None] * len(files)
    height = volume.shape[2]
    max_lines = int(meta.get("max_lines", height))
    code = volume[-1, ..., OCC] > 0.5
    now = np.where(code, risk[-1], 0.0)
    ranked_files, ranges = [], []
    for name, (x0, x1), length in zip(files, bands, lengths):
        rows = now[:, :, x0:x1].max(axis=(0, 2))
        present = code[:, :, x0:x1].any(axis=(0, 2))
        if not present.any():
            continue
        values = rows[present]
        score = float(0.5 * values.max() + 0.5 * values.mean())
        fix_rows = int((volume[-1, :, :, x0:x1, SCAR] > 0.5).any(axis=(0, 2)).sum())
        ranked_files.append({"file": name, "score": score, "fix_rows": fix_rows})
        hot = [int(y) for y in np.where(present & (rows >= max(0.6 * values.max(), 1e-6)))[0]]
        run: list[int] = []
        for y in hot + [None]:
            if y is not None and (not run or y == run[-1] + 1):
                run.append(y)
                continue
            if run:
                first, last = _rows_to_lines(run, height, max_lines)
                if length:
                    first, last = min(first, length), min(last, length)
                ranges.append({"file": name, "lines": [first, last], "score": float(rows[run].max())})
            run = [y] if y is not None else []
    ranked_files.sort(key=lambda item: -item["score"])
    ranges.sort(key=lambda item: -item["score"])
    return {"files": ranked_files[:top], "hotspots": ranges[:top]}


def hotspot_table(volume: np.ndarray, meta: dict, risk: np.ndarray | None = None, top: int = 12) -> list[dict]:
    """Return the ranked list of hotspot line ranges at the newest slice."""
    field = geometric_risk(volume) if risk is None else risk
    return hotspots(volume, meta, field, top=top)["hotspots"]


def report(volume: np.ndarray, meta: dict, model=None, top: int = 12) -> dict:
    """Risk map, ranked files and lines, and a backtest on this history's own fixes."""
    patch = model.cfg.patch if model is not None else 4
    risk = model_risk(model, volume) if model is not None else geometric_risk(volume)
    labels, valid = risk_targets(volume, patch)
    backtest = {"heuristic": risk_scores(heuristic_token_scores(volume, patch), labels, valid)}
    if model is not None and tuple(volume.shape[:4]) == model.cfg.shape:
        import torch

        from tesseract.train import volumes_to_torch

        with torch.no_grad():
            logits = model.risk_logits(volumes_to_torch(volume[None], next(model.parameters()).device))
        backtest["model"] = risk_scores(logits[0].float().cpu().numpy(), labels, valid)
    out = {"source": "model" if model is not None else "heuristic", "backtest": backtest, "weights": dict(WEIGHTS)}
    out.update(hotspots(volume, meta, risk, top=top))
    out["risk"] = risk
    return out


def _clean(value):
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value


def report_text(result: dict, name: str) -> str:
    lines = [f"Fix risk for {name} at its newest commit ({result['source']})."]
    for label, scores in result["backtest"].items():
        auc = scores["auroc"]
        lines.append(
            f"  backtest {label:9s}: AUROC {auc:.3f}  AP {scores['average_precision']:.3f}  "
            f"base rate {scores['base_rate']:.3f}  over {scores['tokens']} patches"
            if auc == auc
            else f"  backtest {label:9s}: this history has no later fixes to score against"
        )
    lines.append("Files most likely to need the next fix:")
    for item in result["files"]:
        lines.append(f"  {item['score']:.2f}  {item['file']}")
    lines.append("Line ranges:")
    for item in result["hotspots"]:
        lines.append(f"  {item['score']:.2f}  {item['file']}:{item['lines'][0]}-{item['lines'][1]}")
    return "\n".join(lines)


def repo_volumes(
    repos: list[str],
    shape: tuple[int, int, int, int],
    windows: int = 3,
    max_commits: int = 600,
) -> list[np.ndarray]:
    """Each repository at several cut points of its history, so one repo gives several solids."""
    from tesseract.codeblock import _git, list_commits, rasterize_repo

    volumes = []
    for repo in repos:
        top = Path(_git(Path(repo), "rev-parse", "--show-toplevel").strip())
        n = len(list_commits(top, "HEAD", max_commits))
        offsets = sorted({int(n * k / (windows + 1)) for k in range(windows)})
        for offset in offsets:
            if n - offset < shape[0]:
                continue
            volume, _inv, _meta = rasterize_repo(top, shape, ref=f"HEAD~{offset}", max_commits=max_commits)
            volumes.append(volume)
            print(f"{top.name} at HEAD~{offset}: {int((volume[..., SCAR] > 0.5).any(axis=(1, 2, 3)).sum())} slices with fixes", flush=True)
    return volumes


def train_risk(
    cfg,
    volumes: list[np.ndarray],
    *,
    steps: int,
    device="cpu",
    lr: float | None = None,
    batch_size: int = 2,
    seed: int = 0,
    init: str | None = None,
    log_every: int = 20,
):
    """Train the causal risk head, with masked reconstruction keeping the encoder honest."""
    import torch
    import torch.nn.functional as F

    from tesseract.model import Eternalist
    from tesseract.train import load_checkpoint, sample_patch_mask, volumes_to_torch

    device = torch.device(device)
    torch.manual_seed(seed)
    if init:
        model, loaded, _blob = load_checkpoint(init, device)
        if loaded.shape != cfg.shape:
            raise ValueError(f"{init} is a {loaded.shape} net; the risk volumes are {cfg.shape}")
        cfg = loaded
    else:
        model = Eternalist(cfg).to(device)
    targets = [risk_targets(volume, cfg.patch) for volume in volumes]
    positives = sum(int((labels & valid).sum()) for labels, valid in targets)
    negatives = sum(int((~labels & valid).sum()) for labels, valid in targets)
    pos_weight = torch.tensor(float(np.clip(negatives / max(positives, 1), 1.0, 50.0)), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr or cfg.lr, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(seed)
    model.train()
    history = []
    for step in range(1, steps + 1):
        pick = rng.integers(0, len(volumes), size=batch_size)
        batch = volumes_to_torch(np.stack([volumes[int(i)] for i in pick]), device)
        labels = torch.from_numpy(np.stack([targets[int(i)][0] for i in pick])).to(device)
        valid = torch.from_numpy(np.stack([targets[int(i)][1] for i in pick])).to(device)
        mask = sample_patch_mask(model, batch_size, cfg.mask_ratio, device)
        output = model(batch, patch_mask=mask, causal=True)
        if bool(valid.any()):
            risk = F.binary_cross_entropy_with_logits(
                output.risk[valid], labels[valid].float(), pos_weight=pos_weight
            )
        else:
            risk = output.risk.sum() * 0.0
        per_token = (output.patch_pred - output.patch_target).pow(2).mean(dim=-1)
        recon = (per_token * mask.float()).sum() / mask.float().sum().clamp(min=1.0)
        loss = risk + 0.5 * recon
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        history.append(float(risk.detach()))
        if log_every and (step % log_every == 0 or step == 1 or step == steps):
            print(f"risk step {step:5d}  risk {float(risk.detach()):.4f}  recon {float(recon.detach()):.4f}", flush=True)
    return model, history


def _load(ckpt: str | None):
    if not ckpt:
        return None
    from tesseract.train import load_checkpoint, pick_device

    model, _cfg, _blob = load_checkpoint(ckpt, pick_device("auto"))
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict where the next fix lands in a repository's solid")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("scan", "backtest"):
        command = sub.add_parser(name)
        command.add_argument("repo", nargs="?", default=".")
        command.add_argument("--ckpt", default=None)
        command.add_argument("--max-commits", type=int, default=600)
        command.add_argument("--top", type=int, default=12)
        command.add_argument("--json", action="store_true")
    train = sub.add_parser("train")
    train.add_argument("repos", nargs="*")
    train.add_argument("--gallery", nargs="*", default=[], help="Gallery sculptures to clone and train on")
    train.add_argument("--config", default="repo", choices=("tiny", "repo", "rtx16gb", "a100"))
    train.add_argument("--init", default=None, help="Start from an existing checkpoint of the same shape")
    train.add_argument("--steps", type=int, default=300)
    train.add_argument("--batch", type=int, default=2)
    train.add_argument("--windows", type=int, default=3)
    train.add_argument("--max-commits", type=int, default=600)
    train.add_argument("--out", default="checkpoints/risk.pt")
    train.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.command == "train":
        from tesseract.config import preset
        from tesseract.eval import evaluate_risk
        from tesseract.sculpture import fetch
        from tesseract.train import pick_device, save_checkpoint

        cfg = preset(args.config)
        repos = list(args.repos) + [str(fetch(name)) for name in args.gallery]
        if not repos:
            parser.error("give at least one repository or --gallery name")
        volumes = repo_volumes(repos, cfg.shape, windows=args.windows, max_commits=args.max_commits)
        if not volumes:
            parser.error(f"no history was long enough for {cfg.shape[0]} time slices")
        device = pick_device(args.device)
        model, _history = train_risk(cfg, volumes, steps=args.steps, device=device, batch_size=args.batch, init=args.init)
        save_checkpoint(args.out, model, args.steps, "risk")
        print(json.dumps(_clean(evaluate_risk(model, volumes, device)), indent=2))
        print(f"wrote {args.out}")
        return

    from tesseract.codeblock import rasterize_repo

    model = _load(args.ckpt)
    shape = model.cfg.shape if model is not None else DEFAULT_SHAPE
    volume, _inv, meta = rasterize_repo(args.repo, shape, max_commits=args.max_commits)
    result = report(volume, meta, model, top=args.top)
    result.pop("risk")
    if args.json:
        print(json.dumps(_clean(result), indent=2))
    elif args.command == "backtest":
        print(json.dumps(_clean(result["backtest"]), indent=2))
    else:
        print(report_text(result, meta["name"]))


if __name__ == "__main__":
    main()
