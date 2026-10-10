"""Slice, interior, invariant, and rotation scores for a block checkpoint."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from tesseract.geom import rotate_xt
from tesseract.life import load_iris, random_spec, rasterize
from tesseract.model import Eternalist
from tesseract.train import load_checkpoint, pick_device, sample_interior, volumes_to_torch


def living_bins(model: Eternalist, invariants: np.ndarray) -> tuple[int, int]:
    lifespan = float(invariants[0])
    alive = []
    for index in range(model.n_t):
        start = index / model.n_t
        end = (index + 1) / model.n_t
        if end > 0.02 and start < max(lifespan, 0.05):
            alive.append(index)
    if not alive:
        alive = [0]
    return alive[0], alive[-1]


@torch.no_grad()
def slice_mae(
    model: Eternalist,
    volume: np.ndarray,
    axis: str,
    index: int,
    *,
    causal: bool,
    device: torch.device,
) -> float:
    model.eval()
    tensor = volumes_to_torch(volume[None], device)
    mask = model.slice_mask(1, axis, index, device)
    output = model(tensor, patch_mask=mask, causal=causal)
    hidden = mask[0]
    error = (output.patch_pred[0, hidden] - output.patch_target[0, hidden]).abs().mean()
    return float(error)


@torch.no_grad()
def evaluate(
    model: Eternalist,
    volumes: list[np.ndarray],
    invariants: list[np.ndarray],
    device: torch.device | str,
    causal: bool = False,
) -> dict[str, float]:
    device = torch.device(device)
    model.eval()
    model.to(device)
    early, late, side, interior, inv_err, cosine = [], [], [], [], [], []
    latents = []
    for volume, inv in zip(volumes, invariants):
        early_bin, late_bin = living_bins(model, inv)
        early.append(slice_mae(model, volume, "t", early_bin, causal=causal, device=device))
        late.append(slice_mae(model, volume, "t", late_bin, causal=causal, device=device))
        side.append(slice_mae(model, volume, "x", model.n_x // 2, causal=causal, device=device))
        tensor = volumes_to_torch(volume[None], device)
        points, values = sample_interior(tensor, model.cfg.interior_queries)
        output = model(tensor, points=points, causal=causal)
        interior.append(float((output.interior - values).abs().mean()))
        target = torch.from_numpy(np.asarray(inv, dtype=np.float32)).to(device)
        valid = target > -0.5
        inv_err.append(float((output.invariants[0, valid] - target[valid]).abs().mean()))
        turned = rotate_xt(tensor, 20.0)
        other = model.latent_only(turned, causal=causal)
        cosine.append(float(F.cosine_similarity(output.latent, other, dim=-1).mean()))
        latents.append(output.latent.detach())
    stacked = torch.cat(latents, dim=0)
    std = float(stacked.std(dim=0).mean()) if stacked.shape[0] > 1 else float(stacked.std())
    early_m = float(np.mean(early))
    late_m = float(np.mean(late))
    return {
        "early_mae": early_m,
        "late_mae": late_m,
        "gap_early_minus_late": early_m - late_m,
        "x_slice_mae": float(np.mean(side)),
        "interior_mae": float(np.mean(interior)),
        "invariant_mae": float(np.mean(inv_err)),
        "rotation_cosine": float(np.mean(cosine)),
        "latent_std": std,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a 4D block checkpoint")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--causal-ckpt", default=None)
    parser.add_argument("--lives", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    device = pick_device(args.device)
    model, cfg, blob = load_checkpoint(args.ckpt, device)
    rng = np.random.default_rng(args.seed)
    specs = [load_iris()] + [random_spec(rng) for _ in range(max(args.lives - 1, 0))]
    volumes, invariants = [], []
    for spec in specs[: args.lives]:
        volume, inv, _meta = rasterize(spec, cfg.shape)
        volumes.append(volume)
        invariants.append(inv)
    causal = blob.get("kind") == "causal"
    metrics = evaluate(model, volumes, invariants, device, causal=causal)
    print(json.dumps({"ckpt": args.ckpt, "kind": blob.get("kind"), **metrics}, indent=2))
    if args.causal_ckpt:
        twin, _cfg, _blob = load_checkpoint(args.causal_ckpt, device)
        twin_metrics = evaluate(twin, volumes, invariants, device, causal=True)
        print(json.dumps({"ckpt": args.causal_ckpt, "kind": "causal", **twin_metrics}, indent=2))


if __name__ == "__main__":
    main()
