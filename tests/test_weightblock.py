import json
import struct
import threading
from urllib.request import Request, urlopen

import numpy as np

from tesseract.demo import make_server
from tesseract.life import N_CHANNELS
from tesseract.weightblock import is_weight_source, read_safetensors, rasterize_weights, weights_probe_sentence


def _write_safetensors(path, tensors):
    header, blobs, offset = {}, [], 0
    for name, array in tensors.items():
        raw = np.ascontiguousarray(array, dtype=np.float32).tobytes()
        header[name] = {"dtype": "F32", "shape": list(array.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    text = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(text)) + text + b"".join(blobs))


def _tensors():
    rng = np.random.default_rng(0)
    w = rng.normal(0, 0.02, (64, 48)).astype(np.float32)
    w[3, 5] = 1.0
    return {"embed.weight": w, "embed.bias": np.zeros(64, np.float32), "head.weight": rng.normal(0, 0.1, (10, 64))}


def test_safetensors_round_trip(tmp_path):
    path = tmp_path / "tiny.safetensors"
    tensors = _tensors()
    _write_safetensors(path, tensors)
    loaded = read_safetensors(path)
    assert set(loaded) == set(tensors)
    np.testing.assert_allclose(loaded["embed.weight"], tensors["embed.weight"])
    assert is_weight_source(str(path)) and is_weight_source("tesseract") and not is_weight_source("x.npz")


def test_weights_become_a_block_with_stats():
    volume, invariants, meta = rasterize_weights(_tensors(), (6, 8, 16, 16), name="tiny")
    assert volume.shape == (6, 8, 16, 16, N_CHANNELS)
    assert meta["kind"] == "weights" and meta["n_tensors"] == 3
    assert meta["n_params"] == 64 * 48 + 64 + 640
    embed = meta["tensors"][0]
    assert embed["shape"] == [64, 48] and embed["outliers"] > 0
    assert volume[..., 3].max() == 1.0
    assert "tiny" in meta["reading"]
    sentence = weights_probe_sentence(meta, {"occupancy": 1.0, "tissue": 1.0, "affect": 0.9, "scar": 1.0}, 0.5, 0.5, 0.9, 0.0)
    assert "embed.weight" in sentence and "outlier" in sentence


def test_viewer_holds_a_weight_file(tmp_path):
    path = tmp_path / "tiny.safetensors"
    _write_safetensors(path, _tensors())
    server, world = make_server("127.0.0.1", 0, shape=(6, 8, 16, 16))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        request = Request(
            f"http://127.0.0.1:{port}/api/open",
            data=json.dumps({"path": str(path)}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request) as response:
            assert json.load(response)["kind"] == "weights"
        with urlopen(f"http://127.0.0.1:{port}/api/block") as response:
            payload = json.load(response)
        assert len(payload["tensors"]) == 3 and payload["points"]
        with urlopen(f"http://127.0.0.1:{port}/api/probe?x=0.5&y=0.5&z=0.1&t=0") as response:
            assert "embed" in json.load(response)["sentence"]
    finally:
        server.shutdown()
