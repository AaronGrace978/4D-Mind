"""A neural network's weights, rasterized as one 4D block.

The axes and channels match a life, so the viewer and the block net read it unchanged.

Axes of the returned array are `(t, z, y, x, channel)`.

    t  depth through the network: tensors in state-dict order, input layers first
    z  magnitude: |w| in units of the tensor's own std, small weights low, outliers high
    y  input unit (columns of the 2D view of the tensor)
    x  output unit (rows of the 2D view of the tensor)

A tensor with more than two dims is viewed as `(shape[0], prod(shape[1:]))`; a vector is one row.
Each tensor is block-averaged onto the y-x grid, keeping the sign of the strongest weight in a cell.
When there are more tensors than t slices, neighbors share a slice and the stronger cell wins.

Channels
    0 weight: a cell holds a weight
    1 positive: the cell's dominant weight is positive
    2 magnitude: |w| relative to the largest |w| of its tensor
    3 outlier: |w| beyond three standard deviations of its tensor
    4 negative: the cell's dominant weight is negative

Sources: `.safetensors` (read with numpy, no extra dependency), `weights:<file>.npz` of named arrays,
`.pt` / `.pth` / `.bin` state dicts (needs torch), a sharded Hugging Face folder or `*.index.json`,
or `tesseract` for this repo's own block net. Safetensors are memory-mapped. A PEFT adapter's
lora_A / lora_B pairs are folded into their update B·A. `tuned - base` holds what training changed.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path

import numpy as np

from tesseract.life import AFFECT, INVARIANT_NAMES, N_CHANNELS, OCC, OTHER, SCAR, TISSUE, invariants_from_volume

WEIGHT_SUFFIXES = (".safetensors", ".pt", ".pth", ".bin", ".ckpt")
LABELS = {
    "occupancy": "weight",
    "tissue": "positive",
    "affect": "magnitude",
    "scar": "outlier",
    "other": "negative",
}
# |w| / std at or past this lands on the top z layer.
MAX_SIGMA = 4.0
OUTLIER_SIGMA = 3.0
STAT_SAMPLE = 1 << 20
SAFETENSOR_DTYPES = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}


def is_weight_source(source: str) -> bool:
    lowered = source.strip().lower()
    if " - " in lowered:
        return all(is_weight_source(part) for part in lowered.split(" - ", 1))
    if lowered == "tesseract" or lowered.startswith("weights:") or lowered.endswith(WEIGHT_SUFFIXES):
        return True
    if lowered.endswith(".index.json"):
        return True
    path = Path(source.strip())
    return path.is_dir() and any(path.glob("*.safetensors"))


class Bf16:
    """A memory-mapped BF16 tensor, widened to float32 only when it is read."""

    def __init__(self, bits: np.ndarray) -> None:
        self.bits = bits
        self.shape = bits.shape
        self.size = bits.size
        self.dtype = np.dtype(np.float32)

    def __array__(self, dtype=None, copy=None):
        out = (self.bits.astype(np.uint32) << 16).view(np.float32)
        return out if dtype is None else out.astype(dtype, copy=False)


def read_safetensors(path: str | Path) -> dict[str, np.ndarray]:
    """Memory-map every tensor; nothing is read until a tensor is used."""
    path = Path(path)
    with path.open("rb") as handle:
        (size,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(size))
    base = 8 + size
    out: dict[str, np.ndarray] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        dtype = info["dtype"]
        shape = tuple(info["shape"])
        if end <= start:
            continue
        if dtype == "BF16":
            out[name] = Bf16(np.memmap(path, dtype=np.uint16, mode="r", offset=base + start, shape=shape))
        elif dtype in SAFETENSOR_DTYPES:
            out[name] = np.memmap(path, dtype=SAFETENSOR_DTYPES[dtype], mode="r", offset=base + start, shape=shape)
    return out


def read_sharded(path: str | Path) -> dict[str, np.ndarray]:
    """A Hugging Face checkpoint split across shards: an index json or a folder of .safetensors."""
    path = Path(path)
    if path.is_dir():
        indexes = sorted(path.glob("*.safetensors.index.json"))
        if not indexes:
            out: dict[str, np.ndarray] = {}
            for shard in sorted(path.glob("*.safetensors")):
                out.update(read_safetensors(shard))
            return out
        path = indexes[0]
    weight_map = json.loads(path.read_text())["weight_map"]
    shards: dict[str, dict] = {}
    out = {}
    for name, shard in weight_map.items():
        if shard not in shards:
            shards[shard] = read_safetensors(path.parent / shard)
        if name in shards[shard]:
            out[name] = shards[shard][name]
    return out


def fold_lora(tensors: dict[str, np.ndarray], scale: float = 1.0) -> dict[str, np.ndarray]:
    """Replace each lora_A / lora_B pair with its update B @ A, the shape of the weight it edits."""
    out: dict[str, np.ndarray] = {}
    for name, value in tensors.items():
        if ".lora_A" in name:
            partner = name.replace(".lora_A", ".lora_B")
            if partner in tensors:
                a = np.asarray(value, dtype=np.float32)
                b = np.asarray(tensors[partner], dtype=np.float32)
                key = name.replace(".lora_A", "").replace(".weight", "") + ".delta"
                out[key] = scale * (b.reshape(b.shape[0], -1) @ a.reshape(a.shape[0], -1))
                continue
        if ".lora_B" in name and name.replace(".lora_B", ".lora_A") in tensors:
            continue
        out[name] = value
    return out


def _lora_scale(folder: Path) -> float:
    config = folder / "adapter_config.json"
    if not config.is_file():
        return 1.0
    blob = json.loads(config.read_text())
    rank = float(blob.get("r") or 0)
    return float(blob.get("lora_alpha", rank)) / rank if rank else 1.0


def _torch_state(path: str | Path) -> dict[str, np.ndarray]:
    import torch

    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(blob, torch.nn.Module):
        blob = blob.state_dict()
    if isinstance(blob, dict):
        for key in ("model", "state_dict", "model_state", "module"):
            if isinstance(blob.get(key), dict):
                blob = blob[key]
                break
    if not isinstance(blob, dict):
        raise ValueError(f"{path} does not hold a state dict")
    return {
        str(name): value.detach().float().cpu().numpy()
        for name, value in blob.items()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    }


def _own_net() -> dict[str, np.ndarray]:
    import torch

    from tesseract.config import TrainConfig
    from tesseract.model import Eternalist

    torch.manual_seed(0)
    model = Eternalist(TrainConfig())
    return {name: value.detach().float().numpy() for name, value in model.state_dict().items()}


def load_weights(source: str | Path) -> tuple[dict[str, np.ndarray], str]:
    text = str(source).strip()
    if text.lower().startswith("weights:"):
        text = text[len("weights:") :].strip()
    if text.lower() == "tesseract":
        return _own_net(), "Eternalist (untrained)"
    path = Path(text)
    if path.is_dir() or path.name.endswith(".index.json"):
        tensors = read_sharded(path)
        name = (path if path.is_dir() else path.parent).name
    elif not path.is_file():
        raise FileNotFoundError(f"no weight file at {path}")
    elif path.suffix == ".safetensors":
        tensors = read_safetensors(path)
        name = path.name
    elif path.suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        tensors = {key: data[key] for key in data.files}
        name = path.name
    else:
        tensors = _torch_state(path)
        name = path.name
    if any(".lora_A" in key for key in tensors):
        folder = path if path.is_dir() else path.parent
        tensors = fold_lora(tensors, _lora_scale(folder))
        name += " (LoRA B·A)"
    if not tensors:
        raise ValueError(f"no float tensors in {path}")
    return tensors, name


def diff_weights(
    tuned: dict[str, np.ndarray], base: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Tuned minus base for every tensor the two share, and each change's norm relative to the base."""
    delta: dict[str, np.ndarray] = {}
    relative: dict[str, float] = {}
    for name, value in tuned.items():
        if name not in base or tuple(np.shape(base[name])) != tuple(np.shape(value)):
            continue
        before = np.asarray(base[name], dtype=np.float32)
        change = np.asarray(value, dtype=np.float32) - before
        delta[name] = change
        norm = float(np.linalg.norm(before.reshape(-1)))
        relative[name] = float(np.linalg.norm(change.reshape(-1))) / norm if norm > 0 else 0.0
    if not delta:
        raise ValueError("the two checkpoints share no tensor of the same shape")
    return delta, relative


def _as_matrix(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 0:
        return array.reshape(1, 1)
    if array.ndim == 1:
        return array.reshape(1, -1)
    return array.reshape(array.shape[0], -1)


def _pool(matrix: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Max-|w| pool a matrix onto at most `rows` x `cols` cells, keeping the sign."""
    out_rows, out_cols = min(rows, matrix.shape[0]), min(cols, matrix.shape[1])
    r_edges = np.linspace(0, matrix.shape[0], out_rows + 1).astype(int)[:-1]
    c_edges = np.linspace(0, matrix.shape[1], out_cols + 1).astype(int)[:-1]
    high = np.maximum.reduceat(np.maximum.reduceat(matrix, r_edges, axis=0), c_edges, axis=1)
    low = np.minimum.reduceat(np.minimum.reduceat(matrix, r_edges, axis=0), c_edges, axis=1)
    return np.where(np.abs(high) >= np.abs(low), high, low).astype(np.float32)


LAYER_PATTERN = re.compile(r"(?:^|\.)(?:layers|blocks|layer|h|block)\.(\d+)\.")


def layer_groups(names: list[str]) -> list[int]:
    """Group index per tensor: one group per numbered layer, one per stray tensor, in order."""
    groups: list[int] = []
    seen: dict[str, int] = {}
    for name in names:
        match = LAYER_PATTERN.search(name)
        key = f"layer {int(match.group(1))}" if match else f"tensor {name}"
        if key not in seen:
            seen[key] = len(seen)
        groups.append(seen[key])
    return groups


def tensor_stats(name: str, array: np.ndarray) -> dict:
    """Exact moments and range; tail shares come from an even stride of at most STAT_SAMPLE values."""
    values = np.asarray(array, dtype=np.float32).reshape(-1)
    count = values.size
    mean = float(values.sum(dtype=np.float64)) / count
    square = float(np.dot(values, values))
    std = float(np.sqrt(max(square / count - mean * mean, 0.0)))
    sample = values[:: max(1, count // STAT_SAMPLE)].astype(np.float64)
    if std > 0:
        z = (sample - mean) / std
        outliers = float((np.abs(z) > OUTLIER_SIGMA).mean())
        kurtosis = float((z**4).mean() - 3.0)
    else:
        outliers, kurtosis = 0.0, 0.0
    return {
        "name": name,
        "shape": [int(n) for n in np.shape(array)],
        "params": int(count),
        "mean": mean,
        "std": std,
        "min": float(values.min()),
        "max": float(values.max()),
        "abs_mean": float(np.abs(sample).mean()),
        "l2": float(np.sqrt(square)),
        "near_zero": float((np.abs(sample) < 1e-3).mean()),
        "outliers": outliers,
        "kurtosis": kurtosis,
    }


def rasterize_weights(
    tensors: dict[str, np.ndarray],
    shape: tuple[int, int, int, int] = (24, 32, 32, 32),
    name: str = "weights",
    relative: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    time, depth, height, width = shape
    names = [key for key, value in tensors.items() if np.size(value) and np.issubdtype(np.dtype(value.dtype), np.number)]
    if not names:
        raise ValueError("no numeric tensors to hold")
    groups = layer_groups(names)
    n_groups = max(groups) + 1
    volume = np.zeros((time, depth, height, width, N_CHANNELS), dtype=np.float32)
    strength = np.full((time, height, width), -1.0, dtype=np.float32)
    stats: list[dict] = []
    for key, group in zip(names, groups):
        array = np.asarray(tensors[key], dtype=np.float32)
        if not np.isfinite(array).all():
            array = np.nan_to_num(array)
        info = tensor_stats(key, array)
        ti = min(int(group * time / n_groups), time - 1)
        info["t"] = ti
        info["group"] = int(group)
        if relative is not None:
            info["rel_change"] = float(relative.get(key, 0.0))
        stats.append(info)
        pooled = _pool(_as_matrix(array), width, height)
        rows, cols = pooled.shape
        x0, y0 = (width - rows) // 2, (height - cols) // 2
        std = info["std"] or 1.0
        peak = max(abs(info["min"]), abs(info["max"])) or 1.0
        sigma = (np.abs(pooled - info["mean"]) / std).T
        values = pooled.T
        region = strength[ti, y0 : y0 + cols, x0 : x0 + rows]
        wins = np.abs(values) > region
        if not wins.any():
            continue
        region[wins] = np.abs(values)[wins]
        yy, xx = np.nonzero(wins)
        zz = np.minimum((sigma[yy, xx] / MAX_SIGMA * depth).astype(int), depth - 1)
        won = values[yy, xx]
        volume[ti, :, yy + y0, xx + x0] = 0.0
        cells = (ti, zz, yy + y0, xx + x0)
        volume[cells + (OCC,)] = 1.0
        volume[cells + (AFFECT,)] = np.abs(won) / peak
        volume[cells + (TISSUE,)] = (won > 0).astype(np.float32)
        volume[cells + (OTHER,)] = (won < 0).astype(np.float32)
        volume[cells + (SCAR,)] = (sigma[yy, xx] > OUTLIER_SIGMA).astype(np.float32)
    invariants = invariants_from_volume(volume)
    total = sum(item["params"] for item in stats)
    meta = {
        "name": name,
        "kind": "weights",
        "shape": list(shape),
        "labels": dict(LABELS),
        "invariants": [float(v) for v in invariants],
        "invariant_names": list(INVARIANT_NAMES),
        "tensors": stats,
        "n_tensors": len(stats),
        "n_params": int(total),
        "n_layers": int(n_groups),
        "diff": relative is not None,
    }
    meta["reading"] = weights_reading(meta)
    return volume, invariants, meta


def hold_weights(
    source: str, shape: tuple[int, int, int, int], base: str | None = None
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Hold one checkpoint, or `tuned - base` when a base is given (or the source reads `a - b`)."""
    if base is None and " - " in source:
        source, base = (part.strip() for part in source.split(" - ", 1))
    tensors, name = load_weights(source)
    relative = None
    if base:
        base_tensors, base_name = load_weights(base)
        tensors, relative = diff_weights(tensors, base_tensors)
        name = f"{name} − {base_name}"
    volume, invariants, meta = rasterize_weights(tensors, shape, name=name, relative=relative)
    meta["root"] = None
    meta["source"] = str(source)
    meta["base"] = base
    return volume, invariants, meta


def _fmt_params(n: int) -> str:
    for unit, size in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= size:
            return f"{n / size:.2f}{unit}"
    return str(n)


def weights_reading(meta: dict) -> str:
    tensors = meta["tensors"]
    spread = [item for item in tensors if item["std"] > 0 and item["params"] > 1] or tensors
    heaviest = max(tensors, key=lambda item: item["params"])
    heavy_tail = max(spread, key=lambda item: item["kurtosis"])
    sparsest = max(tensors, key=lambda item: item["near_zero"])
    if meta.get("diff"):
        lead = (
            f"{meta['name']}: what training changed, {meta['n_tensors']} shared tensors, "
            f"{_fmt_params(meta['n_params'])} parameters. Every dot is a change, not a weight. "
        )
    else:
        lead = f"{meta['name']} as one solid: {meta['n_tensors']} tensors, {_fmt_params(meta['n_params'])} parameters. "
    text = (
        lead
        + f"t walks through {meta['n_layers']} layers from input to output, x is the output unit, y the input unit, "
        "and z lifts each value by how many standard deviations it sits from its tensor's mean. "
        f"The heaviest tensor is {heaviest['name']} {heaviest['shape']}. "
        f"{heavy_tail['name']} has the heaviest tails (excess kurtosis {heavy_tail['kurtosis']:.1f}, "
        f"{heavy_tail['outliers']:.2%} past 3σ); "
        f"{sparsest['name']} is the most nearly zero ({sparsest['near_zero']:.1%} under 1e-3)."
    )
    if meta.get("diff"):
        moved = sorted(tensors, key=lambda item: item.get("rel_change", 0.0), reverse=True)[:3]
        text += " Moved most, relative to the base: " + ", ".join(
            f"{item['name']} ({item['rel_change']:.2%})" for item in moved
        ) + "."
    return text


def tensor_at(meta: dict, t: float) -> list[dict]:
    time = int(meta["shape"][0])
    ti = min(max(int(round(float(t) * max(time - 1, 0))), 0), time - 1)
    return [item for item in meta.get("tensors", []) if item["t"] == ti]


def weights_probe_sentence(meta: dict, channels: dict, x: float, y: float, z: float, t: float) -> str:
    here = tensor_at(meta, t)
    if not here:
        return f"Point ({x:.2f}, {y:.2f}, {z:.2f}, t={t:.2f}) sits between layers. No tensor lives at this depth."
    lines = []
    for item in here[:4]:
        lines.append(
            f"{item['name']} {item['shape']}: {_fmt_params(item['params'])} params, "
            f"mean {item['mean']:+.4f}, std {item['std']:.4f}, range [{item['min']:+.3f}, {item['max']:+.3f}], "
            f"{item['near_zero']:.1%} near zero, {item['outliers']:.2%} outliers, kurtosis {item['kurtosis']:.1f}"
            + (f", changed {item['rel_change']:.2%} of base norm" if "rel_change" in item else "")
        )
    if len(here) > 4:
        lines.append(f"... and {len(here) - 4} more tensors share this slice")
    if channels.get("occupancy", 0) > 0.45:
        sign = "positive" if channels.get("tissue", 0) > 0.45 else "negative"
        cell = f"A {sign} weight at {channels.get('affect', 0):.0%} of its tensor's peak, {z * MAX_SIGMA:.1f}σ out"
        if channels.get("scar", 0) > 0.45:
            cell += ", an outlier"
        lines.insert(0, cell + ".")
    else:
        lines.insert(0, f"Point ({x:.2f}, {y:.2f}, {z:.2f}, t={t:.2f}) is empty at this magnitude.")
    return " ".join(lines[:1]) + "\n" + "\n".join(lines[1:])


def main() -> None:
    parser = argparse.ArgumentParser(description="Rasterize a model's weights into a 4D block")
    parser.add_argument("source", help=".safetensors, .pt/.pth/.bin, .npz of arrays, or 'tesseract'")
    parser.add_argument("--base", default=None, help="Show source minus this checkpoint instead of raw weights")
    parser.add_argument("--out", default="weights_block.npz")
    parser.add_argument("--shape", type=int, nargs=4, default=[24, 32, 32, 32], metavar=("T", "Z", "Y", "X"))
    parser.add_argument("--stats", action="store_true", help="Print per-tensor stats as JSON")
    args = parser.parse_args()
    volume, invariants, meta = hold_weights(args.source, tuple(args.shape), base=args.base)
    np.savez_compressed(args.out, volume=volume, invariants=invariants, meta=np.asarray(json.dumps(meta)))
    print(meta["reading"])
    if args.stats:
        print(json.dumps(meta["tensors"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
