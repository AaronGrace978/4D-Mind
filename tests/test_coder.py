import json
from pathlib import Path

from tesseract.coder import ask, excerpt, source_context
from tesseract.ollama_cloud import default_model, order_models


def test_coding_models_sort_ahead_of_the_rest():
    models = order_models(["gemma4:31b", "kimi-k2.7-code", "gpt-oss:120b", "kimi-k2.7-code"])
    assert [row["name"] for row in models][:2] == ["gpt-oss:120b", "kimi-k2.7-code"]
    assert default_model(models) == "kimi-k2.7-code"
    assert models[-1]["name"] == "gemma4:31b"


def test_excerpt_stays_inside_the_checkout(tmp_path):
    root = tmp_path / "repo"
    nested = root / "src"
    nested.mkdir(parents=True)
    (nested / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    text = excerpt(root, "src/app.py")
    assert text is not None
    assert "def run()" in text
    assert excerpt(root, "../outside.py") is None
    outside = tmp_path / "outside.py"
    outside.write_text("secret\n", encoding="utf-8")
    assert excerpt(root, "../outside.py") is None


def test_coder_sends_the_source_and_refuses_a_solid_without_files(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "parse.py").write_text("def parse(text):\n    return text.split()\n", encoding="utf-8")
    meta = {"kind": "repo", "name": "toy", "reading": "toy as one solid.", "files": ["parse.py"]}
    seen = {}

    def fake_chat(model, messages, key):
        seen["model"] = model
        seen["key"] = key
        seen["user"] = messages[-1]["content"]
        return "Return an empty list when text is empty."

    reply = ask(
        meta,
        root,
        "Fix empty input.",
        "kimi-k2.7-code",
        "test-key",
        hotspots=[{"file": "parse.py", "lines": [1, 2]}],
        chat=fake_chat,
    )
    assert reply["files"] == ["parse.py"]
    assert "def parse" in seen["user"]
    assert seen["model"] == "kimi-k2.7-code"
    assert seen["key"] == "test-key"
    block, files = source_context(None, meta, [], None)
    assert files == []
    assert "no source" in block.lower()
    try:
        ask({"kind": "sculpture", "name": "Doom"}, None, "fix it", "kimi-k2.7-code", "test-key", chat=fake_chat)
        raise AssertionError("a sculpture has no checkout")
    except Exception as exc:  # noqa: BLE001
        assert "no source" in str(exc).lower()


def test_models_endpoint_reports_whether_a_key_is_set():
    import threading
    from urllib.request import urlopen

    from tesseract.demo import make_server

    server, world = make_server("127.0.0.1", 0, shape=(8, 8, 16, 16))
    world.ollama_key = ""
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        with urlopen(f"http://127.0.0.1:{port}/api/models") as response:
            payload = json.load(response)
        assert payload["key_set"] is False
        assert any(row["name"] == "kimi-k2.7-code" for row in payload["models"])
        assert "Coder" in urlopen(f"http://127.0.0.1:{port}/").read().decode("utf-8")
    finally:
        server.shutdown()


def test_path_escape_uses_the_resolved_root(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    assert excerpt(root, "src/../../etc/passwd") is None


def test_fourd_code_lens_and_grounded_coder_need_no_api_key(tmp_path: Path):
    from tesseract.codeblock import ambient_sculpture
    from tesseract.coder import fourd_code_lens
    from tesseract.risk import geometric_risk, hotspot_table

    volume, _inv, meta = ambient_sculpture("doom", (8, 8, 16, 16), seed=3)
    field = geometric_risk(volume)
    hotspots = hotspot_table(volume, meta, field)
    lens = fourd_code_lens(volume, meta, hotspots=hotspots)
    assert lens["n_commits"] >= 8
    assert len(lens["worldlines"]) > 0
    assert "4D Spatiotemporal Code Lens" in lens["summary"]

    # Grounded 4D coder works on a sculpture or a repo without an API key
    reply = ask(
        meta,
        None,
        "Which worldlines have the highest fix-scar density?",
        "grounded:4d-lens",
        None,
        hotspots=hotspots,
        volume=volume,
    )
    assert reply["model"] == "grounded:4d-lens"
    assert "4D Spatiotemporal Code Audit" in reply["reply"]
    assert "lens" in reply

    # Local Ollama model prefix routes without requiring a cloud API key
    local_seen = {}

    def fake_local_chat(model, messages, key):
        local_seen["model"] = model
        local_seen["key"] = key
        local_seen["user"] = messages[-1]["content"]
        return "Decouple co-mutating modules along t."

    root = tmp_path / "repo"
    root.mkdir()
    (root / "engine.py").write_text("def step(dt):\n    return dt\n", encoding="utf-8")
    repo_meta = {"kind": "repo", "name": "engine", "reading": "engine solid.", "files": ["engine.py"], "bands": [[0, 16]]}
    local_reply = ask(
        repo_meta,
        root,
        "Refactor co-churn.",
        "local:deepseek-coder-v2:16b",
        None,
        hotspots=[{"file": "engine.py", "lines": [1, 2], "risk": 0.72}],
        volume=volume,
        chat=fake_local_chat,
    )
    assert local_seen["model"] == "local:deepseek-coder-v2:16b"
    assert "4D Spatiotemporal Code Lens" in local_seen["user"]
    assert "Decouple" in local_reply["reply"]


def test_ablation_lens_and_self_validating_harness(tmp_path: Path):
    import subprocess
    from tesseract.codeblock import ambient_sculpture
    from tesseract.coder import ablation_lens, run_harness, topology_metrics

    volume, _inv, meta = ambient_sculpture("doom", (8, 8, 16, 16), seed=5)
    ablation = ablation_lens(volume, meta, cut_t=0.4)
    assert "4D Ablation Lens" in ablation["summary"]
    assert 0.0 <= ablation["ablation_gain"] <= 1.0
    topo = topology_metrics(volume)
    assert set(topo.keys()) == {"mean_risk", "peak_risk", "indent_load", "scar_load"}

    # Build a small git repo with a passing test, then test read, patch+auto_verify, and rollback_on_fail
    repo = tmp_path / "app"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (repo / "test_calc.py").write_text(
        "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )

    read_res = run_harness(repo, action="read", file="calc.py")
    assert read_res["ok"] is True
    assert "def add" in read_res["content"]

    # Valid refactor that passes pytest
    patch_ok = run_harness(
        repo,
        action="patch",
        file="calc.py",
        find_text="return a + b",
        replace_text="return int(a) + int(b)",
        auto_verify=True,
        rollback_on_fail=True,
    )
    assert patch_ok["ok"] is True
    assert patch_ok["rolled_back"] is False
    assert "PASSED" in patch_ok["output"]
    assert "int(a)" in (repo / "calc.py").read_text(encoding="utf-8")

    # Broken refactor that fails pytest and automatically rolls back
    patch_bad = run_harness(
        repo,
        action="patch",
        file="calc.py",
        find_text="return int(a) + int(b)",
        replace_text="return a - b",
        auto_verify=True,
        rollback_on_fail=True,
    )
    assert patch_bad["ok"] is False
    assert patch_bad["rolled_back"] is True
    assert "rolled back" in patch_bad["output"]
    assert "return int(a) + int(b)" in (repo / "calc.py").read_text(encoding="utf-8")

