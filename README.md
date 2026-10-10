# 4D-Mind

An AI whose input is not a timeline. A life is one 4D solid: every side, the interior, and every moment are coordinates on the same block. The network is a non-causal transformer over that volume. Time is an axis, the same way left and right are axes. Training turns the block in the plane that mixes time with x, so “later” and “to the right” are the same kind of direction.

The heavy run is a notebook for a Colab Pro A100 with high RAM. This repo is the model, that notebook, and a viewer for the solid.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AaronGrace978/4D-Mind/blob/main/notebooks/4d_mind_a100.ipynb)

## What gets trained

1. **Pretrain** the 4D net on procedural lives. It reconstructs held-out slices (a future instant and a far side of the body are the same kind of task), probes interior points, and regresses whole-block facts: how long the tube is, where a scar opens and stays, where a second worldline touches.
2. **Fine-tune** that net on [Iris](lives/iris.json). Her childhood room and her deathbed are the same coordinates. Those moments are neighbors along time the way two rooms are neighbors along x.
3. **LoRA-fine-tune Qwen**. The block latent becomes soft-prompt tokens. On a 40 GB A100 the notebook uses Qwen2.5-3B-Instruct. On an 80 GB A100 it steps up to Qwen2.5-7B-Instruct. The utterances it learns are written from the simulator’s own measurements, so the voice is glued to the volume.
4. A **causal twin** of the same architecture, masked so a token cannot see a later time. At a cut through the block, that witness has not met events on the far side of the cut. The resident has.

Checkpoints land in `MyDrive/4d-mind/` (`eternalist.pt`, `iris_resident.pt`, `causal_twin.pt`, `qwen_block_lora/`, `iris_block.npz`).

## Colab

Open the notebook above, pick a GPU runtime, and set one line:

```python
PRESET = "session"  # one sitting
# PRESET = "long"   # longer pretrain
```

`session` is the short run. `long` multiplies the step counts. The notebook checks for a GPU, mounts Drive, trains all three stages, prints eval scores, exports Iris, and opens a Gradio view: rotate time into x, cut the solid, read the resident and the witness. Open the `gradio.live` link printed in that cell. The `*.prod.colab.dev` address Colab builds for port 7860 returns 404.

With `RESUME = True` (the default), any stage whose checkpoint is already in `MyDrive/4d-mind/` is skipped, so after a disconnect, **Runtime > Run all** continues from the stage that did not finish. The install cell removes Colab's preinstalled `torchao`, because current `peft` refuses to build LoRA layers next to it.

The notebook is generated. Edit `notebooks/build_notebook.py`, then run `python3 notebooks/build_notebook.py`.

## Local viewer

The page draws the worldtube without loading Qwen. Speech stays on the Colab GPU.

```bash
pip install numpy
python3 -m tesseract.demo
```

Then open `http://127.0.0.1:8765`. Drag to orbit, move **Rotate time into x**, drag the cutting plane, and click a point to read the interior. To view the block the notebook exported:

```bash
python3 -m tesseract.demo --npz iris_block.npz
```

## Tests

```bash
pip install -r requirements.txt
python3 -m pytest
```

The CPU tests use the `tiny` preset. They check Iris’s geometry (the two ends of the tube share a room, the scar persists, tissue sits inside the body), that a causal mask cannot see the future, and that a resident who can see the rest of the block reconstructs a hidden early slice that the witness cannot.

## Layout

| Path | Role |
| --- | --- |
| `tesseract/life.py` | Procedural volumes and the Iris rasterizer |
| `tesseract/model.py` | 4D patch transformer, slice / point / invariant heads, soft-prompt bridge |
| `tesseract/train.py` | Pretrain, and the causal twin |
| `tesseract/finetune.py` | Adapt the block net onto Iris |
| `tesseract/lora_voice.py` | LoRA stage |
| `tesseract/eval.py` | Early vs late slice error, interior error, rotation cosine |
| `tesseract/demo.py`, `web/` | Local viewer |
| `tesseract/gradio_app.py` | Colab instrument |
| `notebooks/4d_mind_a100.ipynb` | The A100 entry point, written by `notebooks/build_notebook.py` |

`python3 -m tesseract.train --config a100` and `python3 -m tesseract.finetune --ckpt checkpoints/eternalist.pt` are the same functions the notebook calls.
