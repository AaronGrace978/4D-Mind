"""Rasterize a life into a dense spacetime volume.

Axes of the returned array are `(t, z, y, x, channel)`.

Channels
    0 occupancy of the body
    1 interior tissue (the heart), absent from the skin
    2 affect, painted through the occupied interior
    3 scar filament, present from the opening time through the end of the tube
    4 the second person's worldline

Positions in a spec are normalized `(x, y, z)` in `[0, 1]`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OCC, TISSUE, AFFECT, SCAR, OTHER = range(5)
CHANNEL_NAMES = ("occupancy", "tissue", "affect", "scar", "other")
N_CHANNELS = 5

INVARIANT_NAMES = (
    "lifespan",
    "scar_t",
    "meet_t",
    "scar_x",
    "scar_y",
    "scar_z",
    "meet_x",
    "meet_y",
    "meet_z",
    "extent",
)

# Local body parts in units of the grow-scale. The anchor is the torso center.
# (name, center xyz, radii xyz, channel)
_PARTS = (
    ("torso", (0.00, 0.02, 0.00), (0.10, 0.16, 0.07), OCC),
    ("head", (0.00, 0.24, 0.00), (0.07, 0.08, 0.07), OCC),
    ("arm_l", (-0.16, 0.05, 0.00), (0.045, 0.13, 0.045), OCC),
    ("arm_r", (0.16, 0.05, 0.00), (0.045, 0.13, 0.045), OCC),
    ("leg_l", (-0.05, -0.20, 0.00), (0.045, 0.14, 0.045), OCC),
    ("leg_r", (0.05, -0.20, 0.00), (0.045, 0.14, 0.045), OCC),
    ("heart", (0.025, 0.06, 0.02), (0.035, 0.045, 0.035), TISSUE),
)
_SCAR_LOCAL = np.array([-0.16, 0.00, 0.02], dtype=np.float64)

_NAMES = (
    "Iris",
    "Jun",
    "Nim",
    "Sol",
    "Ara",
    "Kei",
    "Paz",
    "Ona",
    "Ren",
    "Vesper",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def iris_path() -> Path:
    return repo_root() / "lives" / "iris.json"


def load_spec(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return normalize_spec(json.load(handle))


def load_iris() -> dict:
    return load_spec(iris_path())


def normalize_spec(spec: dict) -> dict:
    out = dict(spec)
    out.setdefault("name", "Iris")
    out.setdefault("other_name", "Jun")
    out.setdefault("story", "")
    out.setdefault("seed", 0)
    out.setdefault("lifespan", 0.9)
    out.setdefault("room", [0.5, 0.46, 0.5])
    out.setdefault("return_to_room", True)
    out.setdefault("wander", 0.12)
    out.setdefault("scar_t", 0.4)
    out.setdefault("scar_radius", 0.04)
    out.setdefault("meet_t", 0.55)
    out.setdefault("meet_mode", "alongside")
    out.setdefault("grow", [0.32, 0.62])
    out.setdefault("affect_keyframes", [[0.0, 0.8], [0.5, 0.3], [1.0, 0.5]])
    out["room"] = [float(v) for v in out["room"]]
    out["grow"] = [float(v) for v in out["grow"]]
    out["lifespan"] = float(np.clip(out["lifespan"], 0.5, 1.0))
    out["scar_t"] = float(np.clip(out["scar_t"], 0.05, out["lifespan"] - 0.05))
    out["meet_t"] = float(np.clip(out["meet_t"], 0.15, out["lifespan"] - 0.08))
    out["wander"] = float(np.clip(out["wander"], 0.0, 0.28))
    return out


def random_spec(rng: np.random.Generator) -> dict:
    lifespan = float(rng.uniform(0.78, 0.98))
    room = [float(rng.uniform(0.38, 0.62)) for _ in range(3)]
    room[1] = float(rng.uniform(0.42, 0.55))
    return normalize_spec(
        {
            "name": str(rng.choice(_NAMES)),
            "other_name": str(rng.choice(_NAMES)),
            "seed": int(rng.integers(0, 1_000_000)),
            "lifespan": lifespan,
            "room": room,
            "return_to_room": True,
            "wander": float(rng.uniform(0.06, 0.2)),
            "scar_t": float(rng.uniform(0.22, 0.7)),
            "scar_radius": float(rng.uniform(0.03, 0.05)),
            "meet_t": float(rng.uniform(0.3, 0.72)),
            "meet_mode": "alongside",
            "grow": [0.3, float(rng.uniform(0.55, 0.7))],
            "affect_keyframes": [
                [0.0, float(rng.uniform(0.6, 0.95))],
                [0.35, float(rng.uniform(0.15, 0.5))],
                [0.7, float(rng.uniform(0.1, 0.4))],
                [0.9, float(rng.uniform(0.35, 0.75))],
            ],
        }
    )


def anchor_at(spec: dict, t_frac: float) -> np.ndarray:
    """Body anchor `(x, y, z)` in normalized coordinates."""
    room = np.asarray(spec["room"], dtype=np.float64)
    life = float(spec["lifespan"])
    if (not spec.get("return_to_room", True)) or t_frac <= 0.16 or t_frac >= life - 0.10:
        return room.copy()
    span = max((life - 0.10) - 0.16, 1e-3)
    u = (t_frac - 0.16) / span
    envelope = np.sin(np.pi * np.clip(u, 0.0, 1.0))
    ang = 2.0 * np.pi * u * 1.5 + float(spec["seed"]) * 0.001
    wander = float(spec["wander"])
    offset = wander * envelope * np.array([np.cos(ang), 0.2 * np.sin(2 * ang), np.sin(ang)])
    return np.clip(room + offset, 0.22, 0.78)


def scale_at(spec: dict, t_frac: float) -> float:
    g0, g1 = spec["grow"]
    u = float(np.clip(t_frac / max(spec["lifespan"], 1e-3), 0.0, 1.0))
    return float(g0 + (g1 - g0) * u)


def affect_at(spec: dict, t_frac: float) -> float:
    keys = sorted((float(t), float(v)) for t, v in spec["affect_keyframes"])
    if t_frac <= keys[0][0]:
        return keys[0][1]
    if t_frac >= keys[-1][0]:
        return keys[-1][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t0 <= t_frac <= t1:
            span = max(t1 - t0, 1e-6)
            mix = (t_frac - t0) / span
            return float(v0 + (v1 - v0) * mix)
    return keys[-1][1]


def _voxel_radii(rx: float, ry: float, rz: float, width: int, height: int, depth: int) -> tuple[float, float, float]:
    """Convert normalized radii to voxels, with a floor so tiny grids still hold a body."""
    return (
        max(rz * (depth - 1), 1.15),
        max(ry * (height - 1), 1.15),
        max(rx * (width - 1), 1.15),
    )


def _paint_ellipsoid(
    frame: np.ndarray,
    channel: int,
    zz: np.ndarray,
    yy: np.ndarray,
    xx: np.ndarray,
    center_zyx: tuple[float, float, float],
    radii_zyx: tuple[float, float, float],
) -> None:
    cz, cy, cx = center_zyx
    rz, ry, rx = radii_zyx
    dz = (zz - cz) / max(rz, 1e-3)
    dy = (yy - cy) / max(ry, 1e-3)
    dx = (xx - cx) / max(rx, 1e-3)
    mask = (dx * dx + dy * dy + dz * dz) <= 1.0
    slab = frame[..., channel]
    np.maximum(slab, mask.astype(np.float32), out=slab)


def _to_index(x: float, y: float, z: float, w: int, h: int, d: int) -> tuple[float, float, float]:
    return (x * (w - 1), y * (h - 1), z * (d - 1))


def rasterize(
    spec: dict, shape: tuple[int, int, int, int]
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return `(volume, invariants, meta)` for one life."""
    spec = normalize_spec(spec)
    t_size, depth, height, width = shape
    volume = np.zeros((t_size, depth, height, width, N_CHANNELS), dtype=np.float32)
    zz, yy, xx = np.ogrid[0:depth, 0:height, 0:width]
    times = (np.arange(t_size, dtype=np.float64) + 0.5) / t_size

    for index, t_frac in enumerate(times):
        if t_frac > spec["lifespan"]:
            continue
        frame = volume[index]
        scale = scale_at(spec, float(t_frac))
        anchor = anchor_at(spec, float(t_frac))
        _paint_body(frame, anchor, scale, zz, yy, xx, width, height, depth, OCC)
        if t_frac >= spec["scar_t"]:
            scar_norm = anchor + _SCAR_LOCAL * scale
            sx, sy, sz = _to_index(scar_norm[0], scar_norm[1], scar_norm[2], width, height, depth)
            radius = float(spec["scar_radius"])
            _paint_ellipsoid(
                frame,
                SCAR,
                zz,
                yy,
                xx,
                (sz, sy, sx),
                _voxel_radii(radius, radius, radius, width, height, depth),
            )
        affect = affect_at(spec, float(t_frac))
        occupied = frame[..., OCC] > 0
        frame[..., AFFECT] = np.maximum(frame[..., AFFECT], occupied.astype(np.float32) * affect)
        _paint_other(frame, spec, float(t_frac), anchor, scale, zz, yy, xx, width, height, depth)

    invariants = invariants_from_volume(volume)
    meta = {
        "name": spec["name"],
        "other_name": spec["other_name"],
        "same_room": bool(spec.get("return_to_room", False)),
        "room": [float(v) for v in spec["room"]],
        "story": spec.get("story", ""),
        "invariants": invariants.tolist(),
        "invariant_names": list(INVARIANT_NAMES),
        "lifespan": float(spec["lifespan"]),
        "scar_t": float(spec["scar_t"]),
        "meet_t": float(spec["meet_t"]),
    }
    return volume, invariants, meta


def _paint_body(
    frame: np.ndarray,
    anchor: np.ndarray,
    scale: float,
    zz: np.ndarray,
    yy: np.ndarray,
    xx: np.ndarray,
    width: int,
    height: int,
    depth: int,
    only_channel: int | None,
) -> None:
    for _name, center, radii, channel in _PARTS:
        if only_channel == OCC and channel not in (OCC, TISSUE):
            continue
        if only_channel not in (None, OCC) and channel != only_channel:
            continue
        local = np.asarray(center, dtype=np.float64) * scale
        point = anchor + local
        ix, iy, iz = _to_index(point[0], point[1], point[2], width, height, depth)
        rx, ry, rz = (np.asarray(radii, dtype=np.float64) * scale)
        _paint_ellipsoid(
            frame,
            channel,
            zz,
            yy,
            xx,
            (iz, iy, ix),
            _voxel_radii(float(rx), float(ry), float(rz), width, height, depth),
        )


def _paint_other(
    frame: np.ndarray,
    spec: dict,
    t_frac: float,
    anchor: np.ndarray,
    scale: float,
    zz: np.ndarray,
    yy: np.ndarray,
    xx: np.ndarray,
    width: int,
    height: int,
    depth: int,
) -> None:
    """A second body, close enough to overlap only near the meeting time."""
    gap = abs(t_frac - float(spec["meet_t"]))
    dist = 0.07 + 0.55 * min(gap / 0.25, 1.0)
    other_anchor = np.clip(anchor + np.array([dist, 0.0, 0.02]), 0.15, 0.85)
    other_scale = scale * 0.72
    for _name, center, radii, channel in _PARTS:
        if channel != OCC:
            continue
        local = np.asarray(center, dtype=np.float64) * other_scale
        point = other_anchor + local
        ix, iy, iz = _to_index(float(point[0]), float(point[1]), float(point[2]), width, height, depth)
        rx, ry, rz = np.asarray(radii, dtype=np.float64) * other_scale
        _paint_ellipsoid(
            frame,
            OTHER,
            zz,
            yy,
            xx,
            (iz, iy, ix),
            _voxel_radii(float(rx), float(ry), float(rz), width, height, depth),
        )


def invariants_from_volume(volume: np.ndarray) -> np.ndarray:
    """Ten normalized facts. Missing events use -1."""
    time, depth, height, width, _channels = volume.shape
    out = np.full(len(INVARIANT_NAMES), -1.0, dtype=np.float32)
    occ = volume[..., OCC]
    counts = occ.reshape(time, -1).sum(axis=1)
    peak = float(counts.max()) if time else 0.0
    alive = counts > max(peak * 0.05, 1.0)
    if peak <= 0 or not np.any(alive):
        out[0] = 0.0
        out[9] = 0.0
        return out
    last = int(np.where(alive)[0][-1])
    out[0] = np.float32((last + 0.5) / time)

    scar = volume[..., SCAR]
    scar_counts = scar.reshape(time, -1).sum(axis=1)
    if float(scar_counts.max()) > 1.0:
        first = int(np.where(scar_counts > scar_counts.max() * 0.25)[0][0])
        out[1] = np.float32((first + 0.5) / time)
        com = _center_of_mass(scar)
        out[3], out[4], out[5] = com

    other = volume[..., OTHER]
    overlap = ((other > 0.3) & (occ > 0.3)).astype(np.float32)
    overlap_counts = overlap.reshape(time, -1).sum(axis=1)
    if float(overlap_counts.max()) >= 1.0:
        first = int(np.where(overlap_counts >= 1.0)[0][0])
        out[2] = np.float32((first + 0.5) / time)
        com = _center_of_mass(overlap)
        out[6], out[7], out[8] = com

    out[9] = np.float32(_extent(occ))
    return out


def truncate_invariants(invariants: np.ndarray, visible_end: float) -> np.ndarray:
    """Hide events that sit beyond `visible_end` on the time axis.

    A causal witness standing at that cut has not met them yet.
    """
    out = np.array(invariants, dtype=np.float32, copy=True)
    if out[0] >= 0:
        out[0] = np.float32(min(float(out[0]), visible_end))
    if out[1] > visible_end:
        out[1] = out[3] = out[4] = out[5] = np.float32(-1.0)
    if out[2] > visible_end:
        out[2] = out[6] = out[7] = out[8] = np.float32(-1.0)
    return out


def _center_of_mass(mass: np.ndarray) -> tuple[float, float, float]:
    """`mass` is `(t, z, y, x)`. Returns normalized `(x, y, z)`."""
    spatial = mass.sum(axis=0)
    total = float(spatial.sum())
    if total <= 0:
        return (-1.0, -1.0, -1.0)
    depth, height, width = spatial.shape
    zz, yy, xx = np.ogrid[0:depth, 0:height, 0:width]
    x = float((spatial * xx).sum() / total) / max(width - 1, 1)
    y = float((spatial * yy).sum() / total) / max(height - 1, 1)
    z = float((spatial * zz).sum() / total) / max(depth - 1, 1)
    return (x, y, z)


def _extent(occ: np.ndarray) -> float:
    spatial = occ.sum(axis=0)
    total = float(spatial.sum())
    if total <= 0:
        return 0.0
    depth, height, width = spatial.shape
    zz, yy, xx = np.ogrid[0:depth, 0:height, 0:width]
    x = float((spatial * xx).sum() / total) / max(width - 1, 1)
    y = float((spatial * yy).sum() / total) / max(height - 1, 1)
    z = float((spatial * zz).sum() / total) / max(depth - 1, 1)
    dx = xx / max(width - 1, 1) - x
    dy = yy / max(height - 1, 1) - y
    dz = zz / max(depth - 1, 1) - z
    rms = (spatial * (dx * dx + dy * dy + dz * dz)).sum() / total
    return float(np.sqrt(rms))


def occupancy_com(volume: np.ndarray, frame: int) -> np.ndarray:
    """Normalized `(x, y, z)` center of occupancy at one instant."""
    occ = volume[frame, ..., OCC]
    total = float(occ.sum())
    if total <= 0:
        return np.zeros(3, dtype=np.float64)
    depth, height, width = occ.shape
    zz, yy, xx = np.ogrid[0:depth, 0:height, 0:width]
    x = float((occ * xx).sum() / total) / max(width - 1, 1)
    y = float((occ * yy).sum() / total) / max(height - 1, 1)
    z = float((occ * zz).sum() / total) / max(depth - 1, 1)
    return np.array([x, y, z], dtype=np.float64)


def alive_frames(volume: np.ndarray) -> np.ndarray:
    counts = volume[..., OCC].reshape(volume.shape[0], -1).sum(axis=1)
    peak = float(counts.max()) if len(counts) else 0.0
    return np.where(counts > max(peak * 0.05, 1.0))[0]


def export_npz(path: str | Path, spec: dict, shape: tuple[int, int, int, int]) -> dict:
    volume, invariants, meta = rasterize(spec, shape)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        volume=volume,
        invariants=invariants,
        meta=np.asarray(json.dumps(meta)),
    )
    return meta


def load_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"].item()))
    return data["volume"], data["invariants"], meta


def smear_along_t(volume: np.ndarray, smear: float) -> np.ndarray:
    """Gaussian-blur every channel along time. `smear` is sigma as a fraction of T."""
    t_size = int(volume.shape[0])
    if t_size < 2 or smear <= 0:
        return np.ascontiguousarray(volume, dtype=np.float32)
    sigma = max(float(smear) * t_size, 1e-3)
    radius = max(1, int(np.ceil(3.0 * sigma)))
    grid = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (grid / sigma) ** 2)
    kernel /= kernel.sum()
    out = np.zeros_like(volume, dtype=np.float32)
    index = np.arange(t_size)
    for shift, weight in zip(grid.astype(np.int64), kernel.astype(np.float32)):
        src = np.clip(index - int(shift), 0, t_size - 1)
        out += weight * volume[src]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _luma(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames)
    if frames.ndim == 4:
        rgb = frames[..., :3].astype(np.float32)
        if float(rgb.max(initial=0.0)) > 1.5:
            rgb = rgb / 255.0
        return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)
    plane = frames.astype(np.float32)
    if float(plane.max(initial=0.0)) > 1.5:
        plane = plane / 255.0
    return plane


def _chroma(frames: np.ndarray) -> np.ndarray | None:
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] < 3:
        return None
    rgb = frames[..., :3].astype(np.float32)
    if float(rgb.max(initial=0.0)) > 1.5:
        rgb = rgb / 255.0
    return np.abs(rgb[..., 0] - rgb[..., 2]).astype(np.float32)


def _resize_plane(plane: np.ndarray, height: int, width: int) -> np.ndarray:
    ys = np.clip(((np.arange(height) + 0.5) * plane.shape[0] / height).astype(np.int64), 0, plane.shape[0] - 1)
    xs = np.clip(((np.arange(width) + 0.5) * plane.shape[1] / width).astype(np.int64), 0, plane.shape[1] - 1)
    return plane[ys[:, None], xs]


def _resample_time(seq: np.ndarray, t_size: int) -> np.ndarray:
    if seq.shape[0] == t_size:
        return seq
    idx = np.clip(((np.arange(t_size) + 0.5) * seq.shape[0] / t_size).astype(np.int64), 0, seq.shape[0] - 1)
    return seq[idx]


def audio_envelope(audio: np.ndarray, t_size: int) -> np.ndarray:
    wave = np.asarray(audio, dtype=np.float32).reshape(-1)
    if wave.size == 0:
        return np.zeros(t_size, dtype=np.float32)
    bins = np.array_split(np.abs(wave), t_size)
    env = np.array([float(bin.mean()) if len(bin) else 0.0 for bin in bins], dtype=np.float32)
    peak = float(env.max()) if env.size else 0.0
    if peak > 0:
        env = env / peak
    return env


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Mono waveform in [-1, 1] and the sample rate."""
    import wave

    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        frames = handle.getnframes()
        width = handle.getsampwidth()
        channels = handle.getnchannels()
        raw = handle.readframes(frames)
    if width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width {width}")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data.astype(np.float32), int(rate)


def rasterize_media(
    frames: np.ndarray,
    shape: tuple[int, int, int, int],
    *,
    audio: np.ndarray | None = None,
    smear: float = 0.16,
    name: str = "smear",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Smear video (and optional audio) into the same 5-channel block a life uses.

    t is the smeared timeline. x and y are the frame. z lifts bright samples
    and audio energy off the picture plane, so a cut or a beat becomes a ridge
    you can turn.
    """
    t_size, depth, height, width = shape
    luma = _luma(frames)
    chroma = _chroma(frames)
    if luma.ndim != 3:
        raise ValueError(f"frames must be (F, H, W) or (F, H, W, C), got {tuple(np.asarray(frames).shape)}")
    resized = np.stack([_resize_plane(plane, height, width) for plane in luma], axis=0)
    timed = _resample_time(resized, t_size)
    chroma_t = None
    if chroma is not None:
        chroma_t = _resample_time(np.stack([_resize_plane(plane, height, width) for plane in chroma], axis=0), t_size)
    delta = np.zeros_like(timed)
    delta[1:] = np.abs(timed[1:] - timed[:-1])
    envelope = audio_envelope(audio, t_size) if audio is not None else timed.reshape(t_size, -1).mean(axis=1)
    volume = np.zeros((t_size, depth, height, width, N_CHANNELS), dtype=np.float32)
    zz = (np.arange(depth, dtype=np.float32) + 0.5) / depth
    for index in range(t_size):
        lift = np.clip(0.22 + 0.55 * timed[index] + 0.18 * envelope[index], 0.08, 0.92)
        z_weight = np.exp(-((zz[:, None, None] - lift[None, :, :]) ** 2) / 0.045)
        occ = timed[index][None, :, :] * z_weight
        volume[index, ..., OCC] = occ
        gy, gx = np.gradient(timed[index])
        edges = np.clip(np.hypot(gx, gy) * 3.0, 0.0, 1.0)
        volume[index, ..., TISSUE] = edges[None, :, :] * z_weight
        volume[index, ..., AFFECT] = envelope[index] * occ
        volume[index, ..., SCAR] = np.clip(delta[index] * 2.4, 0.0, 1.0)[None, :, :] * z_weight
        color = chroma_t[index] if chroma_t is not None else (0.35 + 0.65 * envelope[index]) * timed[index]
        volume[index, ..., OTHER] = np.clip(color, 0.0, 1.0)[None, :, :] * z_weight
    volume = smear_along_t(np.clip(volume, 0.0, 1.0), smear)
    invariants = invariants_from_volume(volume)
    meta = {
        "kind": "media",
        "name": name,
        "other_name": "chroma",
        "same_room": False,
        "smear": float(smear),
        "n_frames": int(luma.shape[0]),
        "has_audio": audio is not None,
        "invariants": invariants.tolist(),
        "invariant_names": list(INVARIANT_NAMES),
        "labels": {
            "occupancy": "light",
            "tissue": "edges",
            "affect": "sound",
            "scar": "cuts",
            "other": "chroma",
        },
    }
    meta["reading"] = media_reading(meta)
    return volume, invariants, meta


def media_reading(meta: dict) -> str:
    inv = [float(v) for v in meta["invariants"]]
    audio = "with its sound stained through the interior" if meta.get("has_audio") else "from picture alone"
    parts = [
        f"{meta['name']} is one smeared solid {audio}. {meta.get('n_frames', 0)} frames occupy "
        f"{inv[0]:.0%} of t after a smear of {float(meta.get('smear', 0)):.2f}.",
    ]
    if inv[1] >= 0:
        parts.append(f"A hard cut opens at t={inv[1]:.2f} and stays as a filament.")
    else:
        parts.append("No hard cut survives the smear as a scar.")
    parts.append(f"The spread of the light through the block is {inv[9]:.2f}.")
    return "\n\n".join(parts)


def media_probe_sentence(channels: dict[str, float], x: float, y: float, z: float, t: float) -> str:
    bits = [f"Smear x={x:.2f} y={y:.2f} z={z:.2f} t={t:.2f}."]
    if channels["occupancy"] > 0.3:
        bits.append("Smeared light occupies this coordinate.")
    else:
        bits.append("This coordinate is between frames.")
    if channels["tissue"] > 0.3:
        bits.append("An edge, streaked through time, is here.")
    if channels["scar"] > 0.3:
        bits.append("A cut or sudden change passes through this point.")
    if channels["other"] > 0.3:
        bits.append("Chroma or a second voice is stained here.")
    bits.append(f"Sound on the solid: {channels['affect']:.2f}.")
    return " ".join(bits)


def synthetic_clip(kind: str, n_frames: int = 24, height: int = 32, width: int = 32, seed: int = 0):
    """Return `(frames, audio)` for a built-in smear. Frames are `(F, H, W)` in 0..1."""
    rng = np.random.default_rng(seed)
    yy, xx = np.ogrid[0:height, 0:width]
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    frames = np.zeros((n_frames, height, width), dtype=np.float32)
    samples = n_frames * 64
    time = np.arange(samples, dtype=np.float32) / 64.0
    if kind == "sweep":
        for index in range(n_frames):
            col = (index / max(n_frames - 1, 1)) * (width - 1)
            frames[index] = np.exp(-((xx - col) ** 2) / 4.5)
        audio = np.sin(2.0 * np.pi * (180.0 + 420.0 * (time / max(time[-1], 1e-3))) * time / n_frames)
    elif kind == "flicker":
        for index in range(n_frames):
            on = 1.0 if index % 3 else 0.12
            frames[index] = on * np.clip(1.0 - np.hypot(yy - cy, xx - cx) / (0.55 * width), 0.0, 1.0)
            if index % 7 == 0:
                frames[index] = np.clip(frames[index] + rng.random((height, width)).astype(np.float32) * 0.8, 0, 1)
        audio = (np.sin(2.0 * np.pi * 8.0 * time / n_frames) > 0).astype(np.float32) * np.sin(2.0 * np.pi * 90.0 * time)
    elif kind == "chord":
        centers = ((0.32, 0.38), (0.68, 0.40), (0.50, 0.68))
        freqs = (180.0, 270.0, 360.0)
        audio = np.zeros(samples, dtype=np.float32)
        for (fy, fx), freq in zip(centers, freqs):
            for index in range(n_frames):
                pulse = 0.45 + 0.55 * np.sin(2.0 * np.pi * (index / max(n_frames, 1)) + fx)
                frames[index] = np.maximum(
                    frames[index],
                    pulse * np.exp(-(((yy - fy * height) ** 2 + (xx - fx * width) ** 2) / 28.0)),
                )
            audio += 0.33 * np.sin(2.0 * np.pi * freq * time / n_frames)
    else:
        kind = "pulse"
        for index in range(n_frames):
            radius = 1.2 + 0.85 * index
            frames[index] = np.exp(-((np.hypot(yy - cy, xx - cx) - radius) ** 2) / 8.0)
        audio = np.sin(2.0 * np.pi * 220.0 * time / n_frames) * np.exp(-time / max(n_frames, 1))
    return frames, audio.astype(np.float32), kind


def rasterize_smear(
    shape: tuple[int, int, int, int],
    kind: str = "pulse",
    *,
    smear: float = 0.16,
    wav: str | Path | None = None,
    frames: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build a smeared media solid from a clip kind, optional WAV, or given frames."""
    audio = None
    name = kind
    if frames is None:
        if kind == "iris-echo":
            volume, invariants, meta = rasterize(load_iris(), shape)
            volume = smear_along_t(volume, smear)
            invariants = invariants_from_volume(volume)
            meta = dict(meta)
            meta.update(
                {
                    "kind": "media",
                    "name": "Iris echo",
                    "smear": float(smear),
                    "n_frames": int(shape[0]),
                    "has_audio": False,
                    "invariants": invariants.tolist(),
                    "labels": {
                        "occupancy": "body",
                        "tissue": "tissue",
                        "affect": "affect",
                        "scar": "scar",
                        "other": "other",
                    },
                }
            )
            meta["reading"] = (
                f"Iris, smeared along t by {smear:.2f}. Childhood and the deathbed bleed into "
                "the frames between them. The tube is still one solid; the cut is thicker."
            )
            return volume, invariants, meta
        frames, audio, name = synthetic_clip(kind, n_frames=max(shape[0], 12), height=shape[2], width=shape[3], seed=seed)
    if wav is not None:
        audio, _rate = load_wav(wav)
        name = Path(wav).stem
    return rasterize_media(frames, shape, audio=audio, smear=smear, name=name)


SMEAR_KINDS = ("pulse", "sweep", "flicker", "chord", "iris-echo")
