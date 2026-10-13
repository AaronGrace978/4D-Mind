import json
import threading
from urllib.request import Request, urlopen

import numpy as np

from tesseract.character import grounded_answer, intent_of
from tesseract.codeblock import ambient_sculpture, list_sculptures
from tesseract.config import repo, tiny
from tesseract.demo import make_server
from tesseract.eval import evaluate_risk
from tesseract.life import OCC, SCAR, rasterize_smear, smear_along_t
from tesseract.model import Eternalist
from tesseract.platform import catalog
from tesseract.risk import geometric_risk, new_fix, risk_targets
from tesseract.voice_text import simultaneous_line


def test_ambient_sculptures_are_distinct_solids():
    names = {item["id"] for item in list_sculptures()}
    assert {"linux", "doom", "git", "cpython"} <= names
    linux, _inv, linux_meta = ambient_sculpture("linux", (8, 8, 16, 16), seed=1)
    doom, _dinv, doom_meta = ambient_sculpture("doom", (8, 8, 16, 16), seed=1)
    assert linux_meta["kind"] == "sculpture"
    assert linux.shape == (8, 8, 16, 16, 5)
    assert linux[..., OCC].sum() > 0
    assert doom[..., OCC].sum() > 0
    assert linux_meta["theme"]["bg"] != doom_meta["theme"]["bg"]


def test_smear_turns_a_clip_into_the_same_block_a_life_uses():
    volume, inv, meta = rasterize_smear((8, 8, 16, 16), "pulse", smear=0.2)
    assert volume.shape == (8, 8, 16, 16, 5)
    assert meta["kind"] == "media"
    assert volume[..., OCC].sum() > 0
    assert inv.shape[0] == 10
    echo, _einv, echo_meta = rasterize_smear((8, 8, 16, 16), "iris-echo", smear=0.25)
    sharp, _sinv, _smeta = rasterize_smear((8, 8, 16, 16), "iris-echo", smear=0.0)
    smeared = smear_along_t(sharp, 0.25)
    assert echo_meta["name"] == "Iris echo"
    assert float(np.abs(echo - smeared).max()) < 1e-5


def test_geometric_risk_is_causal_and_ranks_later_fixes():
    volume, _inv, _meta = ambient_sculpture("doom", (8, 8, 16, 16), seed=2)
    field = geometric_risk(volume)
    assert field.shape == volume.shape[:4]
    born = new_fix(volume)
    if born[1:].any():
        labels, valid = risk_targets(volume, 4)
        assert labels.shape == valid.shape
        assert bool(valid.any())
    earlier = field[0].sum()
    later = field[-1].sum()
    assert later >= earlier * 0.2
    scores = evaluate_risk(None, [volume], "cpu")
    assert "heuristic" in scores
    assert scores["heuristic"]["tokens"] >= 0


def test_risk_head_emits_one_logit_per_token():
    cfg = tiny()
    model = Eternalist(cfg)
    volume = np.zeros((1, cfg.in_ch, *cfg.shape), dtype=np.float32)
    import torch

    logits = model.risk_logits(torch.from_numpy(volume))
    assert logits.shape == (1, model.n_tokens)
    assert repo().shape == (16, 8, 32, 32)


def test_resident_sees_both_sides_of_an_instant():
    _volume, _inv, meta = ambient_sculpture("linux", (8, 8, 16, 16), seed=0)
    assert intent_of("What is ahead of me?") == "ahead"
    line = simultaneous_line(meta, 0.4)
    assert "t=0.40" in line
    answer = grounded_answer(meta, 0.1, "What is ahead of me?")
    assert "ahead" in answer.lower() or "t=" in answer


def test_platform_studios_and_apis(monkeypatch):
    monkeypatch.delenv("FOURD_MIND_LORA", raising=False)
    monkeypatch.delenv("FOURD_MIND_RESIDENT_CKPT", raising=False)
    monkeypatch.delenv("FOURD_MIND_LLM_URL", raising=False)
    platform = catalog()
    assert platform["version"] == "1.2.0"
    assert [studio["id"] for studio in platform["studios"]] == ["sculpt", "scan", "resident", "coder", "smear"]
    server, world = make_server("127.0.0.1", 0, shape=(8, 8, 16, 16), sculpture="doom")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        with urlopen(f"http://127.0.0.1:{port}/api/platform") as response:
            payload = json.load(response)
        assert payload["shell"] == "tauri-2"
        with urlopen(f"http://127.0.0.1:{port}/api/block") as response:
            block = json.load(response)
        assert block["kind"] == "sculpture"
        assert block["name"] == "Doom"
        request = Request(
            f"http://127.0.0.1:{port}/api/resident",
            data=json.dumps({"query": "Who are you, from where I stand?", "t": 0.3}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            reply = json.load(response)
        assert reply["backend"] == "grounded"
        assert reply["text"]
        smear = Request(
            f"http://127.0.0.1:{port}/api/smear",
            data=json.dumps({"kind": "sweep", "smear": 0.12}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(smear) as response:
            assert json.load(response)["kind"] == "media"
        with urlopen(f"http://127.0.0.1:{port}/") as response:
            page = response.read().decode("utf-8")
        assert "Rotate time into x" in page
        assert "Sculpt" in page
        assert world.meta["kind"] == "media"
    finally:
        server.shutdown()
