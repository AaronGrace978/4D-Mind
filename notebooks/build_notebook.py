#!/usr/bin/env python3
"""Write notebooks/4d_mind_a100.ipynb. Edit this file, not the JSON."""

import json
from pathlib import Path

BRANCH = "main"


def cell(kind, source):
    lines = source.strip("\n").split("\n")
    source_lines = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": source_lines}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines,
    }


cells = []

cells.append(cell("markdown", """
# 4D Block Mind

A life is one solid. This notebook pretrains a non-causal 4D transformer on spacetime volumes, fine-tunes it onto Iris, trains a causal witness, then LoRA-tunes Qwen so the block latent is its sense organ.

Use a Colab Pro **A100** runtime with high RAM, then **Runtime > Run all**. Set `PRESET` to `session` for one sitting or `long` for a longer pretrain. Weights go to `MyDrive/4d-mind/`.

With `RESUME = True`, a stage whose checkpoint is already on Drive is skipped. After a disconnect or a restart, Run all picks up where it stopped.
"""))

cells.append(cell("code", """
import torch

if not torch.cuda.is_available():
    raise SystemExit("Select a GPU runtime. This notebook trains on an A100.")
name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"{name}  {vram_gb:.1f} GB")
if "A100" not in name:
    print("This preset is tuned for an A100. A smaller GPU may need batch 1 and can run out of memory.")
"""))

cells.append(cell("code", """
import os
import subprocess
import sys

REPO = "https://github.com/AaronGrace978/4D-Mind.git"
BRANCH = "__BRANCH__"
TARGET = "/content/4D-Mind"

if os.path.exists(os.path.join(TARGET, ".git")):
    # An earlier session's clone is stale; take the branch as it is on GitHub.
    subprocess.check_call(["git", "-C", TARGET, "fetch", "origin", BRANCH])
    subprocess.check_call(["git", "-C", TARGET, "reset", "--hard", "origin/" + BRANCH])
else:
    subprocess.check_call(["git", "clone", "--branch", BRANCH, REPO, TARGET])
os.chdir(TARGET)
if TARGET not in sys.path:
    sys.path.insert(0, TARGET)
print(subprocess.check_output(["git", "log", "-1", "--oneline"], text=True).strip())

# Without a restart, Python keeps running the tesseract code it imported earlier,
# even though the files above were just updated.
stale = [module for module in sys.modules if module == "tesseract" or module.startswith("tesseract.")]
for module in stale:
    del sys.modules[module]
if stale:
    print(f"dropped {len(stale)} tesseract modules imported before this update")
""".replace("__BRANCH__", BRANCH)))

cells.append(cell("markdown", """
Colab ships an old `torchao`. Current `peft` refuses to build LoRA layers while it is installed, and nothing here uses it, so it is removed.
"""))

cells.append(cell("code", """
import importlib.metadata as metadata
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"])
subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])

for package in ("torch", "transformers", "peft", "accelerate", "gradio"):
    print(f"{package:13s}", metadata.version(package))
try:
    metadata.version("torchao")
    raise SystemExit("torchao is still installed. Run: !pip uninstall -y torchao, then Runtime > Restart session.")
except metadata.PackageNotFoundError:
    pass

# A library imported before pip replaced it keeps running the old version.
changed = []
for package in ("transformers", "peft", "accelerate", "gradio"):
    loaded = sys.modules.get(package)
    if loaded is not None and getattr(loaded, "__version__", None) != metadata.version(package):
        changed.append(f"{package} {loaded.__version__} -> {metadata.version(package)}")
if changed:
    raise SystemExit("pip upgraded libraries this session had already imported (" + ", ".join(changed)
                     + "). Runtime > Restart session, then Run all; finished stages are skipped.")
"""))

cells.append(cell("markdown", """
Mount Drive so a disconnect does not throw away the block. If the mount dialog is dismissed, checkpoints fall back to `/content/4d-mind`.
"""))

cells.append(cell("code", """
from pathlib import Path

try:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT = Path("/content/drive/MyDrive/4d-mind")
except Exception as exc:
    print("Drive is not mounted:", exc)
    OUT = Path("/content/4d-mind")

OUT.mkdir(parents=True, exist_ok=True)
print("checkpoints ->", OUT)
for path in sorted(OUT.iterdir()):
    print("  found", path.name)
"""))

cells.append(cell("code", """
PRESET = "session"  # or "long"
RESUME = True       # skip a stage whose checkpoint is already on Drive

import torch
from tesseract.config import a100, a100_runtime

mem_kb = int(open("/proc/meminfo").read().split("MemTotal:")[1].split()[0])
ram_gb = mem_kb / 1024**2
runtime = a100_runtime(vram_gb, ram_gb, PRESET)
cfg = a100()
device = torch.device("cuda")
print("preset", PRESET, " resume", RESUME)
print("language model", runtime["language_model"])
print("batch", runtime["batch"], " cache", runtime["cache"], " steps", runtime["steps"])
print("volume", cfg.shape, " dim", cfg.dim, " depth", cfg.depth, f" RAM {ram_gb:.0f} GB")


def done(name):
    path = OUT / name
    if RESUME and path.exists():
        print(f"{name} is on Drive. Skipping this stage; set RESUME = False to retrain it.")
        return True
    return False


_cache = {}


def training_lives():
    if "lives" not in _cache:
        from tesseract.train import make_cache

        print(f"rasterizing {runtime['cache']} procedural lives", flush=True)
        _cache["lives"] = make_cache(cfg, runtime["cache"], seed=cfg.seed, mode="procedural")
    return _cache["lives"]
"""))

cells.append(cell("markdown", """
## Pretrain

Procedural lives. The loss hides a time slice or an x slice and asks the net to rebuild it, probes points inside the body, and matches whole-block facts. Every few steps it also turns the volume in the time–x plane and asks the latent to stay put.
"""))

cells.append(cell("code", """
from tesseract.model import Eternalist
from tesseract.train import run_training

if not done("eternalist.pt"):
    volumes, invariants, _metas = training_lives()
    torch.manual_seed(cfg.seed)
    model = Eternalist(cfg)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters, {model.n_tokens} tokens")
    run_training(
        model,
        cfg,
        volumes,
        invariants,
        steps=runtime["steps"]["pretrain"],
        lr=cfg.lr,
        device=device,
        causal=False,
        kind="eternalist",
        out_path=OUT / "eternalist.pt",
        batch_size=runtime["batch"],
        seed=cfg.seed,
        log_every=25,
    )
    del model
    torch.cuda.empty_cache()
"""))

cells.append(cell("markdown", """
## Fine-tune on Iris

Her childhood room and her deathbed share coordinates. The block net moves at a lower learning rate on jittered copies of that one life.
"""))

cells.append(cell("code", """
from tesseract.finetune import finetune_iris

if not done("iris_resident.pt"):
    resident, _ = finetune_iris(
        str(OUT / "eternalist.pt"),
        str(OUT / "iris_resident.pt"),
        steps=runtime["steps"]["finetune"],
        device=device,
        cache=min(8, runtime["cache"]),
        batch_size=runtime["batch"],
        seed=cfg.seed,
    )
    del resident
    torch.cuda.empty_cache()
"""))

cells.append(cell("markdown", """
## Causal twin

Same architecture, with a mask that blocks every later time. This witness is trained from scratch so its weights never attended to the future.
"""))

cells.append(cell("code", """
if not done("causal_twin.pt"):
    volumes, invariants, _metas = training_lives()
    torch.manual_seed(cfg.seed + 1)
    twin = Eternalist(cfg)
    run_training(
        twin,
        cfg,
        volumes,
        invariants,
        steps=runtime["steps"]["causal"],
        lr=cfg.lr,
        device=device,
        causal=True,
        kind="causal",
        out_path=OUT / "causal_twin.pt",
        batch_size=runtime["batch"],
        seed=cfg.seed + 1,
        log_every=25,
    )
    del twin
    torch.cuda.empty_cache()
"""))

cells.append(cell("markdown", """
## Eval

Scored on Iris and on lives neither network trained on. `early_mae` is a hidden childhood slice; `late_mae` is a hidden late slice. A resident who can see the whole block should do both. The witness has trouble with the early one, because the future is what would have explained it. `rotation_cosine` compares the latent of a block with the latent of the same block turned 20 degrees in the time–x plane.
"""))

cells.append(cell("code", """
import json

import numpy as np

from tesseract.eval import evaluate
from tesseract.life import load_iris, random_spec, rasterize
from tesseract.train import load_checkpoint

rng = np.random.default_rng(cfg.seed + 1000)
held_out = [load_iris()] + [random_spec(rng) for _ in range(3)]
hold_v, hold_i = [], []
for spec in held_out:
    volume, inv, _meta = rasterize(spec, cfg.shape)
    hold_v.append(volume)
    hold_i.append(inv)

resident, _, _ = load_checkpoint(OUT / "iris_resident.pt", device)
twin, _, _ = load_checkpoint(OUT / "causal_twin.pt", device)
resident_metrics = evaluate(resident, hold_v, hold_i, device, causal=False)
witness_metrics = evaluate(twin, hold_v, hold_i, device, causal=True)
print("resident")
print(json.dumps(resident_metrics, indent=2))
print("witness")
print(json.dumps(witness_metrics, indent=2))
(OUT / "metrics.json").write_text(json.dumps({"resident": resident_metrics, "witness": witness_metrics}, indent=2))
del resident, twin
torch.cuda.empty_cache()
"""))

cells.append(cell("code", """
from tesseract.life import export_npz

export_npz(OUT / "iris_block.npz", load_iris(), cfg.shape)
print("wrote", OUT / "iris_block.npz")
"""))

cells.append(cell("markdown", """
A still of Iris before the language-model stage. Mint is the body, rose is interior tissue, amber is the scar, blue is Jun.
"""))

cells.append(cell("code", """
import matplotlib.pyplot as plt

from tesseract.gradio_app import render_projection, render_slice

preview_volume, _, _ = rasterize(load_iris(), cfg.shape)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
axes[0].imshow(render_projection(preview_volume, 28))
axes[0].set_title("time turned 28° toward x")
axes[1].imshow(render_slice(preview_volume, "t", 0.45))
axes[1].set_title("slice at t = 0.45")
for axis in axes:
    axis.axis("off")
fig.tight_layout()
fig.savefig(OUT / "preview.png", dpi=120)
plt.show()
del preview_volume
"""))

cells.append(cell("markdown", """
## LoRA voice

The encoder stays frozen. Its latent becomes eight soft-prompt tokens, and LoRA adapts Qwen so those tokens carry the solid. Targets are sentences built from the simulator's measurements, not a costume prompt.
"""))

cells.append(cell("code", """
import gc

from tesseract.lora_voice import train_voice

_cache.clear()
gc.collect()
torch.cuda.empty_cache()

if not done("qwen_block_lora/bridge.pt"):
    train_voice(
        encoder_ckpt=str(OUT / "iris_resident.pt"),
        out_dir=str(OUT / "qwen_block_lora"),
        model_name=runtime["language_model"],
        steps=runtime["steps"]["lora"],
        device=device,
        cache=min(16, runtime["cache"]),
        lm_batch=runtime["lm_batch"],
        seed=cfg.seed,
        log_every=10,
    )
gc.collect()
torch.cuda.empty_cache()
"""))

cells.append(cell("markdown", """
## Look

Rotate time into x, cut the block, and ask. The resident answers from the whole solid. The witness answers from the causal twin, pooled only up to the cut. Set `SHARE = True` for a public link that works outside this tab.
"""))

cells.append(cell("code", """
from tesseract.gradio_app import launch

SHARE = False

launch(
    resident_ckpt=str(OUT / "iris_resident.pt"),
    causal_ckpt=str(OUT / "causal_twin.pt"),
    lora_dir=str(OUT / "qwen_block_lora"),
    share=SHARE,
)
"""))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "A100", "machine_shape": "hm"},
    },
    "cells": cells,
}

path = Path(__file__).resolve().with_name("4d_mind_a100.ipynb")
path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
print("wrote", path, "cells", len(cells))
