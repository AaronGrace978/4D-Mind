import json
import subprocess
import threading
from urllib.request import Request, urlopen

import numpy as np

from tesseract.codeblock import rasterize_repo
from tesseract.demo import make_server
from tesseract.life import AFFECT, OCC, OTHER, SCAR, TISSUE

SHAPE = (6, 8, 16, 8)


def _commit(repo, author, message, files):
    for name, text in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if text is None:
            path.unlink()
        else:
            path.write_text(text, encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", f"user.name={author}", "-c", f"user.email={author}@example.com",
         "-c", "commit.gpgsign=false", "commit", "-q", "-m", message],
        check=True,
    )


def _make_repo(tmp_path):
    repo = tmp_path / "toy"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    _commit(repo, "ada", "Start the parser", {"src/parse.py": "def parse(text):\n    return text.split()\n"})
    _commit(repo, "ada", "Add a lexer", {"src/lex.py": "class Lexer:\n    def run(self):\n        return []\n"})
    _commit(
        repo,
        "ada",
        "Fix empty input in parse",
        {"src/parse.py": "def parse(text):\n    if not text:\n        return []\n    return text.split()\n"},
    )
    _commit(repo, "bo", "Document the lexer", {"src/lex.py": "# tokens\nclass Lexer:\n    def run(self):\n        return []\n"})
    _commit(repo, "ada", "Drop notes", {"notes.txt": "not code\n"})
    _commit(repo, "ada", "Remove the lexer", {"src/lex.py": None})
    return repo


def test_repo_history_becomes_a_block(tmp_path):
    repo = _make_repo(tmp_path)
    volume, invariants, meta = rasterize_repo(repo, SHAPE)
    assert volume.shape == SHAPE + (5,)
    assert meta["kind"] == "repo"
    assert meta["n_commits"] == 6
    assert meta["files"] == ["src/lex.py", "src/parse.py"]
    assert meta["main_author"] == "ada"
    assert [entry["subject"] for entry in meta["slices"]][2] == "Fix empty input in parse"

    lex = slice(*meta["bands"][0])
    code = volume[..., OCC]
    assert code[0, :, :, lex].sum() == 0
    assert code[1, :, :, lex].sum() > 0
    assert code[5, :, :, lex].sum() == 0, "a deleted file leaves the solid"

    scar = volume[..., SCAR].reshape(SHAPE[0], -1).sum(axis=1)
    assert scar[:2].sum() == 0
    assert np.all(scar[2:] > 0), "fix lines persist after the fix"
    assert invariants[1] == np.float32(2.5 / 6)

    other = volume[..., OTHER].reshape(SHAPE[0], -1).sum(axis=1)
    assert other[:3].sum() == 0 and other[3] > 0 and other[5] == 0
    assert volume[..., TISSUE].sum() > 0
    assert volume[2, ..., AFFECT].max() > 0
    assert volume[4, ..., AFFECT].max() == 0, "a commit that touches no code does not churn"
    assert "Fix empty input in parse" in meta["reading"]


def test_viewer_opens_a_repo_and_names_the_file(tmp_path):
    repo = _make_repo(tmp_path)
    server, world = make_server("127.0.0.1", 0, shape=SHAPE)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        request = Request(
            f"http://127.0.0.1:{port}/api/open",
            data=json.dumps({"path": str(repo)}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            assert json.load(response)["kind"] == "repo"
        with urlopen(f"http://127.0.0.1:{port}/api/block") as response:
            block = json.load(response)
        assert block["labels"]["scar"] == "fix lines"
        point = next(p for p in block["points"] if p[7] > 0.45)
        query = f"x={point[0]}&y={point[1]}&z={point[2]}&t={point[3]}"
        with urlopen(f"http://127.0.0.1:{port}/api/probe?{query}") as response:
            sentence = json.load(response)["sentence"]
        assert "src/parse.py" in sentence
        assert "fix" in sentence.lower()

        bare = Request(f"http://127.0.0.1:{port}/api/open", data=b"path=/", method="POST")
        try:
            urlopen(bare)
            raise AssertionError("a form post must not swap the solid")
        except Exception as exc:  # noqa: BLE001
            assert getattr(exc, "code", None) == 415
        assert world.meta["kind"] == "repo"
    finally:
        server.shutdown()
