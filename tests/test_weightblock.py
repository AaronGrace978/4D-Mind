import json
import struct
import threading
from urllib.request import Request, urlopen

import numpy as np

from tesseract.demo import make_server
from tesseract.life import N_CHANNELS
from tesseract.weightblock import (
    _pool,
    hold_weights,
    is_weight_source,
    layer_groups,
    load_weights,
    rasterize_weights,
    read_safetensors,
    weights_probe_sentence,
)


def _write_safetensors(path, tensors, bf16=False):
    header, blobs, offset = {}, [], 0
    for name, array in tensors.items():
        floats = np.ascontiguousarray(array, dtype=np.float32)
        raw = (floats.view(np.uint32) >> 16).astype(np.uint16).tobytes() if bf16 else floats.tobytes()
        header[name] = {"dtype": "BF16" if bf16 else "F32", "shape": list(array.shape), "data_offsets": [offset, offset + len(raw)]}
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


def test_pool_keeps_the_strongest_signed_weight():
    matrix = np.array([[0.1, -0.9, 0.2, 0.3], [0.4, 0.5, -0.1, 0.8]], dtype=np.float32)
    np.testing.assert_allclose(_pool(matrix, 1, 2), [[-0.9, 0.8]])
    np.testing.assert_allclose(_pool(matrix, 8, 8), matrix)


def test_bf16_sharded_folder_and_layer_groups(tmp_path):
    first = {"model.layers.0.q.weight": np.ones((4, 4)), "model.layers.0.k.weight": -np.ones((4, 4))}
    second = {"model.layers.1.q.weight": np.full((4, 4), 0.5), "lm_head.weight": np.ones((2, 4))}
    _write_safetensors(tmp_path / "model-00001-of-00002.safetensors", first, bf16=True)
    _write_safetensors(tmp_path / "model-00002-of-00002.safetensors", second, bf16=True)
    weight_map = {name: "model-00001-of-00002.safetensors" for name in first}
    weight_map.update({name: "model-00002-of-00002.safetensors" for name in second})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    assert is_weight_source(str(tmp_path))
    tensors, _name = load_weights(tmp_path)
    assert len(tensors) == 4
    np.testing.assert_allclose(np.asarray(tensors["model.layers.1.q.weight"]), 0.5)
    assert layer_groups(list(tensors)) == [0, 0, 1, 2]
    _volume, _inv, meta = hold_weights(str(tmp_path), (6, 4, 8, 8))
    assert meta["n_layers"] == 3


def test_lora_pairs_fold_into_their_update(tmp_path):
    a = np.ones((2, 6), np.float32)
    b = np.full((5, 2), 0.5, np.float32)
    _write_safetensors(tmp_path / "adapter_model.safetensors", {"q.lora_A.weight": a, "q.lora_B.weight": b})
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4}))
    tensors, name = load_weights(tmp_path / "adapter_model.safetensors")
    assert "LoRA" in name and list(tensors) == ["q.delta"]
    np.testing.assert_allclose(tensors["q.delta"], 2.0 * (b @ a))


def test_diff_of_two_checkpoints_names_what_moved(tmp_path):
    base = _tensors()
    tuned = {key: value.copy() for key, value in base.items()}
    tuned["head.weight"] = tuned["head.weight"] * 1.5
    _write_safetensors(tmp_path / "base.safetensors", base)
    _write_safetensors(tmp_path / "tuned.safetensors", tuned)
    source = f"{tmp_path / 'tuned.safetensors'} - {tmp_path / 'base.safetensors'}"
    assert is_weight_source(source)
    _volume, _inv, meta = hold_weights(source, (6, 8, 16, 16))
    assert meta["diff"]
    moved = {item["name"]: item["rel_change"] for item in meta["tensors"]}
    assert abs(moved["head.weight"] - 0.5) < 1e-4 and moved["embed.weight"] == 0.0
    assert "head.weight" in meta["reading"].split("Moved most")[1]
