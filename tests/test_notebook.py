import ast
import json
from pathlib import Path


def test_colab_notebook_has_the_training_stages():
    path = Path("notebooks/4d_mind_a100.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    sources = []
    for item in notebook["cells"]:
        sources.append("".join(item["source"]))
    joined = "\n".join(sources)
    assert notebook["nbformat"] == 4
    for phrase in (
        "A100",
        "requirements-colab.txt",
        "cursor/4d-block-mind-d0d3",
        "MyDrive/4d-mind",
        "PRESET",
        "run_training",
        "finetune_iris",
        "causal=True",
        "evaluate",
        "iris_block.npz",
        "train_voice",
        "Qwen",
        "launch",
    ):
        assert phrase in joined, phrase
    # Install after the repo is present, so the requirements file exists.
    clone_at = joined.index('git", "clone')
    install_at = joined.index("requirements-colab.txt")
    assert clone_at < install_at
    for item in notebook["cells"]:
        if item["cell_type"] != "code":
            continue
        ast.parse("".join(item["source"]))
