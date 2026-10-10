"""Pretrain the eternalist, or train the causal witness, on cached lives."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tesseract.config import TrainConfig, config_from_dict, preset
from tesseract.geom import rotate_xt
from tesseract.life import invariants_from_volume, load_iris, random_spec, rasterize, truncate_invariants
from tesseract.model import Eternalist


def volumes_to_torch(volumes: np.ndarray, device: torch.device | str) -> torch.Tensor:
    """`(B, T, D, H, W, C)` numpy -> `(B, C, T, D, H, W)` float tensor."""
    tensor = torch.from_numpy(np.ascontiguousarray(volumes))
    return tensor.permute(0, 5, 1, 2, 3, 4).contiguous().to(device)


def jitter_volume(volume: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = volume
    for axis, shift in enumerate(rng.integers(-1, 2, size=3), start=1):
        out = np.roll(out, int(shift), axis=axis)
    if rng.random() < 0.5:
        noise = rng.normal(0.0, 0.01, size=out.shape).astype(np.float32)
        out = np.clip(out + noise, 0.0, 1.0)
    return np.ascontiguousarray(out, dtype=np.float32)


def make_cache(
    cfg: TrainConfig,
    n: int,
    seed: int,
    mode: str = "procedural",
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict]]:
    rng = np.random.default_rng(seed)
    volumes: list[np.ndarray] = []
    invariants: list[np.ndarray] = []
    metas: list[dict] = []
    iris = load_iris() if mode == "iris" else None
    for index in range(n):
        spec = iris if iris is not None else random_spec(rng)
        volume, inv, meta = rasterize(spec, cfg.shape)
        if mode == "iris" and index > 0:
            volume = jitter_volume(volume, rng)
            inv = invariants_from_volume(volume)
            meta = dict(meta)
            meta["invariants"] = inv.tolist()
        volumes.append(volume)
        invariants.append(inv)
        metas.append(meta)
    return volumes, invariants, metas


def sample_patch_mask(
    model: Eternalist,
    batch: int,
    mask_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    """Hide a whole time or x slice, plus a random sprinkle of patches."""
    tokens = model.n_tokens
    mask = torch.zeros(batch, tokens, dtype=torch.bool, device=device)
    time_index = model.t_index
    x_index = model.patch_index[:, 3]
    for item in range(batch):
        if torch.rand((), device=device).item() < 0.75:
            if torch.rand((), device=device).item() < 0.5:
                chosen = int(torch.randint(0, model.n_t, ()).item())
                mask[item] = time_index == chosen
            else:
                chosen = int(torch.randint(0, model.n_x, ()).item())
                mask[item] = x_index == chosen
        sprinkle = torch.rand(tokens, device=device) < mask_ratio
        mask[item] = mask[item] | sprinkle
        if bool(mask[item].all()):
            mask[item, 0] = False
        if not bool(mask[item].any()):
            mask[item, int(torch.randint(0, tokens, ()).item())] = True
    return mask


def sample_interior(volume: torch.Tensor, queries: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Half the probes land inside the body. Returns coords `(B, Q, 4)` and values `(B, Q, C)`."""
    batch, _channels, time, depth, height, width = volume.shape
    device = volume.device
    coords = torch.empty(batch, queries, 4, device=device)
    flat = volume.permute(0, 2, 3, 4, 5, 1)
    values = torch.empty(batch, queries, volume.shape[1], device=device)
    for item in range(batch):
        inside = (volume[item, 0] > 0.3).nonzero(as_tuple=False)
        n_inside = min(queries // 2, int(inside.shape[0]))
        parts = []
        if n_inside > 0:
            choice = torch.randint(0, inside.shape[0], (n_inside,), device=device)
            parts.append(inside[choice])
        n_rest = queries - (parts[0].shape[0] if parts else 0)
        random_idx = torch.stack(
            [
                torch.randint(0, time, (n_rest,), device=device),
                torch.randint(0, depth, (n_rest,), device=device),
                torch.randint(0, height, (n_rest,), device=device),
                torch.randint(0, width, (n_rest,), device=device),
            ],
            dim=1,
        )
        parts.append(random_idx)
        index = torch.cat(parts, dim=0)
        if index.shape[0] < queries:
            extra = index[torch.randint(0, index.shape[0], (queries - index.shape[0],), device=device)]
            index = torch.cat([index, extra], dim=0)
        index = index[:queries]
        coords[item, :, 0] = (index[:, 0].float() + 0.5) / time
        coords[item, :, 1] = (index[:, 1].float() + 0.5) / depth
        coords[item, :, 2] = (index[:, 2].float() + 0.5) / height
        coords[item, :, 3] = (index[:, 3].float() + 0.5) / width
        values[item] = flat[item, index[:, 0], index[:, 1], index[:, 2], index[:, 3]]
    return coords, values


def rotation_loss(model: Eternalist, volume: torch.Tensor, degrees: float) -> torch.Tensor:
    angle = (torch.rand((), device=volume.device).item() * 2.0 - 1.0) * degrees
    turned = rotate_xt(volume, angle)
    latent = model.latent_only(volume)
    with torch.no_grad():
        other = model.latent_only(turned)
    left = F.normalize(latent, dim=-1)
    right = F.normalize(other, dim=-1)
    return (1.0 - (left * right).sum(dim=-1)).mean()


def _truncated_targets(invariants: np.ndarray, until: torch.Tensor, n_t: int) -> torch.Tensor:
    visible = ((until.detach().cpu().numpy().astype(np.float64) + 1.0) / n_t)
    rows = [truncate_invariants(invariants[i], float(visible[i])) for i in range(len(visible))]
    return torch.from_numpy(np.stack(rows)).to(until.device)


def compute_loss(
    model: Eternalist,
    volume: torch.Tensor,
    invariants: torch.Tensor,
    *,
    causal: bool,
    rotation_degrees: float,
    do_rotation: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    mask = sample_patch_mask(model, volume.shape[0], model.cfg.mask_ratio, volume.device)
    points, point_values = sample_interior(volume, model.cfg.interior_queries)
    until = None
    targets = invariants
    if causal:
        until = torch.randint(0, model.n_t, (volume.shape[0],), device=volume.device)
        targets = _truncated_targets(invariants.detach().cpu().numpy(), until, model.n_t)
    output = model(volume, patch_mask=mask, causal=causal, points=points, until_bin=until)
    per_token = (output.patch_pred - output.patch_target).pow(2).mean(dim=-1)
    recon = (per_token * mask.float()).sum() / mask.float().sum().clamp(min=1.0)
    valid = targets > -0.5
    if bool(valid.any()):
        inv = F.mse_loss(output.invariants[valid], targets[valid])
    else:
        inv = output.invariants.sum() * 0.0
    point = F.mse_loss(output.interior, point_values)
    total = recon + 0.5 * point + 0.5 * inv
    turned = volume.new_zeros(())
    if do_rotation and not causal:
        turned = rotation_loss(model, volume, rotation_degrees)
        total = total + 0.25 * turned
    parts = {
        "loss": float(total.detach()),
        "recon": float(recon.detach()),
        "point": float(point.detach()),
        "inv": float(inv.detach()),
        "rot": float(turned.detach()),
    }
    return total, parts


def run_training(
    model: Eternalist,
    cfg: TrainConfig,
    volumes: list[np.ndarray],
    invariants: list[np.ndarray],
    *,
    steps: int,
    lr: float,
    device: torch.device | str,
    causal: bool = False,
    rotation_every: int | None = None,
    log_every: int = 20,
    kind: str = "eternalist",
    out_path: str | Path | None = None,
    batch_size: int = 2,
    seed: int = 0,
) -> list[float]:
    device = torch.device(device)
    model.to(device)
    model.train()
    params = list(model.perception_parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(seed)
    every = cfg.rotation_every if rotation_every is None else rotation_every
    history: list[float] = []
    use_amp = device.type == "cuda"
    stacked_inv = np.stack(invariants).astype(np.float32)
    for step in range(1, steps + 1):
        choice = rng.integers(0, len(volumes), size=batch_size)
        batch_np = np.stack([volumes[int(i)] for i in choice])
        target_np = stacked_inv[choice]
        batch = volumes_to_torch(batch_np, device)
        target = torch.from_numpy(np.ascontiguousarray(target_np)).to(device)
        opt.zero_grad(set_to_none=True)
        do_rotation = every > 0 and step % every == 0
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, parts = compute_loss(
                    model,
                    batch,
                    target,
                    causal=causal,
                    rotation_degrees=cfg.rotation_degrees,
                    do_rotation=do_rotation,
                )
        else:
            loss, parts = compute_loss(
                model,
                batch,
                target,
                causal=causal,
                rotation_degrees=cfg.rotation_degrees,
                do_rotation=do_rotation,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        history.append(parts["loss"])
        if log_every and (step % log_every == 0 or step == 1 or step == steps):
            print(
                f"{kind} step {step:5d}  loss {parts['loss']:.4f}  recon {parts['recon']:.4f}"
                f"  point {parts['point']:.4f}  inv {parts['inv']:.4f}  rot {parts['rot']:.4f}",
                flush=True,
            )
    if out_path is not None:
        save_checkpoint(out_path, model, steps, kind)
    return history


def save_checkpoint(path: str | Path, model: Eternalist, step: int, kind: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "cfg": model.cfg.to_dict(),
            "step": step,
            "kind": kind,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[Eternalist, TrainConfig, dict]:
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = config_from_dict(blob["cfg"])
    model = Eternalist(cfg)
    missing, unexpected = model.load_state_dict(blob["model"], strict=False)
    # Checkpoints from before the risk head load with a fresh one.
    missing = [key for key in missing if not key.startswith("risk_head.")]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint {path} does not match the model: missing {missing}, unexpected {unexpected}")
    model.to(device)
    return model, cfg, blob


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain a 4D block model")
    parser.add_argument("--config", default="tiny", choices=("tiny", "repo", "a100"))
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--cache", type=int, default=8)
    parser.add_argument("--out", default="checkpoints/eternalist.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    cfg = preset(args.config)
    device = pick_device(args.device)
    print(f"device {device}  preset {cfg.name}  volume {cfg.shape}  causal {args.causal}", flush=True)
    volumes, invariants, _metas = make_cache(cfg, args.cache, args.seed, mode="procedural")
    torch.manual_seed(args.seed)
    model = Eternalist(cfg)
    kind = "causal" if args.causal else "eternalist"
    run_training(
        model,
        cfg,
        volumes,
        invariants,
        steps=args.steps,
        lr=cfg.lr,
        device=device,
        causal=args.causal,
        kind=kind,
        out_path=args.out,
        batch_size=args.batch,
        seed=args.seed,
    )
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
