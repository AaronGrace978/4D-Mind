"""Numpy views of a spacetime block: points, slices, and interior probes."""

from __future__ import annotations

import numpy as np

from tesseract.life import AFFECT, CHANNEL_NAMES, OCC, OTHER, SCAR, TISSUE


def sample_points(volume: np.ndarray, max_points: int = 7000, seed: int = 0) -> np.ndarray:
    """Return `(N, 9)` rows: x, y, z, t, then the five channels. Coordinates are 0–1."""
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    for channel, cap in ((SCAR, 1800), (TISSUE, 1600), (OTHER, 1600)):
        idx = np.argwhere(volume[..., channel] > 0.45)
        if len(idx) > cap:
            idx = idx[rng.choice(len(idx), cap, replace=False)]
        if len(idx):
            chosen.append(idx)
    used = sum(len(item) for item in chosen)
    remain = max_points - used
    occ = np.argwhere(volume[..., OCC] > 0.45)
    if remain > 0 and len(occ):
        if len(occ) > remain:
            occ = occ[rng.choice(len(occ), remain, replace=False)]
        chosen.append(occ)
    if not chosen:
        return np.zeros((0, 4 + len(CHANNEL_NAMES)), dtype=np.float32)
    idx = np.concatenate(chosen, axis=0)
    time, depth, height, width, _channels = volume.shape
    coords = np.stack(
        [
            idx[:, 3] / max(width - 1, 1),
            idx[:, 2] / max(height - 1, 1),
            idx[:, 1] / max(depth - 1, 1),
            idx[:, 0] / max(time - 1, 1),
        ],
        axis=1,
    )
    values = volume[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]]
    return np.concatenate([coords, values], axis=1).astype(np.float32)


def probe_channels(volume: np.ndarray, x: float, y: float, z: float, t: float) -> dict[str, float]:
    values = _sample_trilinear(volume, x, y, z, t)
    return {name: float(values[index]) for index, name in enumerate(CHANNEL_NAMES)}


def slice_image(volume: np.ndarray, axis: str, value: float) -> np.ndarray:
    """RGB uint8 image of one cutting plane through the block."""
    time, depth, height, width, _channels = volume.shape
    value = float(np.clip(value, 0.0, 1.0))
    if axis == "t":
        index = int(round(value * (time - 1)))
        occ = volume[index, ..., OCC]
        z_index = int(occ.sum(axis=(1, 2)).argmax()) if occ.sum() > 0 else depth // 2
        plane = volume[index, z_index]
    elif axis == "z":
        index = int(round(value * (depth - 1)))
        mass = volume[:, index, ..., OCC].reshape(time, -1).sum(axis=1)
        t_index = int(mass.argmax()) if mass.sum() > 0 else 0
        plane = volume[t_index, index]
    elif axis == "y":
        index = int(round(value * (height - 1)))
        mass = volume[:, :, index, :, OCC].reshape(time, -1).sum(axis=1)
        t_index = int(mass.argmax()) if mass.sum() > 0 else 0
        plane = volume[t_index, :, index, :]
    elif axis == "x":
        index = int(round(value * (width - 1)))
        slab = volume[:, :, :, index, :]
        mass = slab[..., OCC].sum(axis=(0, 2))
        z_index = int(mass.argmax()) if mass.sum() > 0 else depth // 2
        plane = slab[:, z_index]
    else:
        raise ValueError(f"unknown axis {axis!r}")
    return _colorize(plane)


def _sample_trilinear(volume: np.ndarray, x: float, y: float, z: float, t: float) -> np.ndarray:
    time, depth, height, width, channels = volume.shape

    def axis_loc(coord: float, size: int) -> tuple[int, int, float]:
        pos = float(np.clip(coord, 0.0, 1.0)) * (size - 1)
        left = int(np.floor(pos))
        right = min(left + 1, size - 1)
        return left, right, pos - left

    t0, t1, ta = axis_loc(t, time)
    z0, z1, za = axis_loc(z, depth)
    y0, y1, ya = axis_loc(y, height)
    x0, x1, xa = axis_loc(x, width)
    acc = np.zeros(channels, dtype=np.float64)
    for ti, tw in ((t0, 1.0 - ta), (t1, ta)):
        for zi, zw in ((z0, 1.0 - za), (z1, za)):
            for yi, yw in ((y0, 1.0 - ya), (y1, ya)):
                for xi, xw in ((x0, 1.0 - xa), (x1, xa)):
                    acc += tw * zw * yw * xw * volume[ti, zi, yi, xi]
    return acc.astype(np.float32)


def _colorize(plane: np.ndarray) -> np.ndarray:
    occ = plane[..., OCC]
    tissue = plane[..., TISSUE]
    affect = plane[..., AFFECT]
    scar = plane[..., SCAR]
    other = plane[..., OTHER]
    rgb = np.zeros(plane.shape[:2] + (3,), dtype=np.float32)
    rgb[..., 0] = np.clip(scar + 0.65 * tissue + 0.15 * occ * affect, 0, 1)
    rgb[..., 1] = np.clip(occ * (0.25 + 0.75 * affect), 0, 1)
    rgb[..., 2] = np.clip(other + 0.35 * tissue + 0.05 * occ, 0, 1)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
