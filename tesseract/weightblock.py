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
`.pt` / `.pth` / `.bin` state dicts (needs torch), or `tesseract` for this repo's own block net.
"""

from __future__ import annotations

import argparse
import json
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
    lowered = source.lower()
    return lowered == "tesseract" or lowered.startswith("weights:") or lowered.endswith(WEIGHT_SUFFIXES)


def read_safetensors(path: str | Path) -> dict[str, np.ndarray]:
    raw = Path(path).read_bytes()
    (size,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + size])
    base = 8 + size
    out: dict[str, np.ndarray] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        chunk = raw[base + start : base + end]
        dtype = info["dtype"]
        if dtype == "BF16":
            bits = np.frombuffer(chunk, dtype=np.uint16).astype(np.uint32) << 16
            array = bits.view(np.float32)
        elif dtype in SAFETENSOR_DTYPES:
            array = np.frombuffer(chunk, dtype=SAFETENSOR_DTYPES[dtype])
        else:
            continue
        out[name] = array.reshape(info["shape"])
    return out


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
    text = str(source)
    if text.lower().startswith("weights:"):
        text = text[len("weights:") :].strip()
    if text.lower() == "tesseract":
        return _own_net(), "Eternalist (untrained)"
    path = Path(text)
    if not path.is_file():
        raise FileNotFoundError(f"no weight file at {path}")
    if path.suffix == ".safetensors":
        tensors = read_safetensors(path)
    elif path.suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        tensors = {name: data[name] for name in data.files}
    else:
        tensors = _torch_state(path)
    return tensors, path.name


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
    r_edges = np.linspace(0, matrix.shape[0], out_rows + 1).astype(int)
    c_edges = np.linspace(0, matrix.shape[1], out_cols + 1).astype(int)
    pooled = np.zeros((out_rows, out_cols), dtype=np.float32)
    for i in range(out_rows):
        band = matrix[r_edges[i] : max(r_edges[i + 1], r_edges[i] + 1)]
        for j in range(out_cols):
            cell = band[:, c_edges[j] : max(c_edges[j + 1], c_edges[j] + 1)]
            flat = cell.reshape(-1)
            pooled[i, j] = flat[int(np.argmax(np.abs(flat)))]
    return pooled


def tensor_stats(name: str, array: np.ndarray) -> dict:
    values = np.asarray(array, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if not len(finite):
        finite = np.zeros(1)
    std = float(finite.std())
    return {
        "name": name,
        "shape": [int(n) for n in np.shape(array)],
        "params": int(values.size),
        "mean": float(finite.mean()),
        "std": std,
        "min": float(finite.min()),
        "max": float(finite.max()),
        "abs_mean": float(np.abs(finite).mean()),
        "l2": float(np.sqrt((finite**2).sum())),
        "near_zero": float((np.abs(finite) < 1e-3).mean()),
        "outliers": float((np.abs(finite - finite.mean()) > OUTLIER_SIGMA * std).mean()) if std > 0 else 0.0,
    }


def rasterize_weights(
    tensors: dict[str, np.ndarray],
    shape: tuple[int, int, int, int] = (24, 32, 32, 32),
    name: str = "weights",
) -> tuple[np.ndarray, np.ndarray, dict]:
    time, depth, height, width = shape
    names = [key for key, value in tensors.items() if np.size(value) and np.issubdtype(np.asarray(value).dtype, np.number)]
    if not names:
        raise ValueError("no numeric tensors to hold")
    volume = np.zeros((time, depth, height, width, N_CHANNELS), dtype=np.float32)
    strength = np.full((time, height, width), -1.0, dtype=np.float32)
    stats: list[dict] = []
    for index, key in enumerate(names):
        array = np.nan_to_num(np.asarray(tensors[key], dtype=np.float32))
        info = tensor_stats(key, array)
        ti = min(int(index * time / len(names)), time - 1)
        info["t"] = ti
        stats.append(info)
        pooled = _pool(_as_matrix(array), width, height)
        rows, cols = pooled.shape
        x0, y0 = (width - rows) // 2, (height - cols) // 2
        std = info["std"] or 1.0
        peak = max(abs(info["min"]), abs(info["max"])) or 1.0
        sigma = np.abs(pooled - info["mean"]) / std
        for r in range(rows):
            for c in range(cols):
                value = float(pooled[r, c])
                xi, yi = x0 + r, y0 + c
                if abs(value) <= strength[ti, yi, xi]:
                    continue
                strength[ti, yi, xi] = abs(value)
                volume[ti, :, yi, xi] = 0.0
                zi = min(int(sigma[r, c] / MAX_SIGMA * depth), depth - 1)
                cell = volume[ti, zi, yi, xi]
                cell[OCC] = 1.0
                cell[AFFECT] = abs(value) / peak
                cell[TISSUE] = 1.0 if value > 0 else 0.0
                cell[OTHER] = 1.0 if value < 0 else 0.0
                cell[SCAR] = 1.0 if sigma[r, c] > OUTLIER_SIGMA else 0.0
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
    }
    meta["reading"] = weights_reading(meta)
    return volume, invariants, meta


def hold_weights(source: str, shape: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray, dict]:
    tensors, name = load_weights(source)
    volume, invariants, meta = rasterize_weights(tensors, shape, name=name)
    meta["root"] = None
    meta["source"] = str(source)
    return volume, invariants, meta


def _fmt_params(n: int) -> str:
    for unit, size in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= size:
            return f"{n / size:.2f}{unit}"
    return str(n)


def weights_reading(meta: dict) -> str:
    tensors = meta["tensors"]
    heaviest = max(tensors, key=lambda item: item["params"])
    wildest = max(tensors, key=lambda item: item["outliers"])
    sparsest = max(tensors, key=lambda item: item["near_zero"])
    return (
        f"{meta['name']} as one solid: {meta['n_tensors']} tensors, {_fmt_params(meta['n_params'])} parameters. "
        "t walks through the network from input to output, x is the output unit, y the input unit, "
        "and z lifts each weight by how many standard deviations it sits from its tensor's mean. "
        f"The heaviest tensor is {heaviest['name']} {heaviest['shape']}. "
        f"{wildest['name']} carries the most outliers ({wildest['outliers']:.1%} past 3σ); "
        f"{sparsest['name']} is the most nearly zero ({sparsest['near_zero']:.1%} under 1e-3)."
    )


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
            f"{item['near_zero']:.1%} near zero, {item['outliers']:.2%} outliers"
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
    parser.add_argument("--out", default="weights_block.npz")
    parser.add_argument("--shape", type=int, nargs=4, default=[24, 32, 32, 32], metavar=("T", "Z", "Y", "X"))
    parser.add_argument("--stats", action="store_true", help="Print per-tensor stats as JSON")
    args = parser.parse_args()
    volume, invariants, meta = hold_weights(args.source, tuple(args.shape))
    np.savez_compressed(args.out, volume=volume, invariants=invariants, meta=np.asarray(json.dumps(meta)))
    print(meta["reading"])
    if args.stats:
        print(json.dumps(meta["tensors"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
