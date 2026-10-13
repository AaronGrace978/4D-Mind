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
        'BRANCH = "main"',
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
        "gradio.live",
        '"uninstall", "-y", "-q", "torchao"',
        "RESUME = True",
        '"reset", "--hard"',
        'module.startswith("tesseract.")',
        "Restart session",
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


def test_notebook_calls_match_the_real_signatures():
    import inspect

    from tesseract.coder import fourd_code_lens, grounded_code_review
    from tesseract.eval import evaluate, evaluate_risk
    from tesseract.finetune import finetune_iris
    from tesseract.gradio_app import launch
    from tesseract.lora_voice import train_voice
    from tesseract.risk import train_risk
    from tesseract.train import run_training, save_checkpoint

    functions = {
        fn.__name__: fn
        for fn in (
            fourd_code_lens,
            grounded_code_review,
            evaluate,
            evaluate_risk,
            finetune_iris,
            launch,
            train_voice,
            train_risk,
            run_training,
            save_checkpoint,
        )
    }
    notebook = json.loads(Path("notebooks/4d_mind_a100.ipynb").read_text(encoding="utf-8"))
    checked = set()
    for item in notebook["cells"]:
        if item["cell_type"] != "code":
            continue
        for node in ast.walk(ast.parse("".join(item["source"]))):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in functions):
                continue
            signature = inspect.signature(functions[node.func.id])
            args = [ast.unparse(arg) for arg in node.args]
            kwargs = {kw.arg: ast.unparse(kw.value) for kw in node.keywords}
            signature.bind(*args, **kwargs)
            checked.add(node.func.id)
    assert {"train_risk", "train_voice", "run_training", "finetune_iris", "launch"} <= checked


def test_notebook_matches_its_generator(tmp_path):
    import shutil
    import subprocess
    import sys

    source = Path("notebooks/build_notebook.py")
    copy = tmp_path / "build_notebook.py"
    shutil.copy(source, copy)
    subprocess.check_call([sys.executable, str(copy)], stdout=subprocess.DEVNULL)
    built = (tmp_path / "4d_mind_a100.ipynb").read_text(encoding="utf-8")
    committed = Path("notebooks/4d_mind_a100.ipynb").read_text(encoding="utf-8")
    assert built == committed, "run python3 notebooks/build_notebook.py"
