# 4D Mind

An AI platform whose input is not a timeline. A life, a repository, or a clip is one 4D solid: every side, the interior, and every moment are coordinates on the same block. The network is a non-causal transformer over that volume. Time is an axis, the same way left and right are axes.

The desktop shell is **Tauri 2.12**. Four studios share one solid:

| Studio | What it is | Where it lives |
| --- | --- | --- |
| **Sculpt** | Ambient 3D code sculptures — Linux, Doom, Git, CPython, Redis, SQLite | `tesseract/codeblock.py` and the local web viewer |
| **Scan** | Geometric code vulnerability predictor — where the next fix is likely to land | `tesseract/model.py`, `tesseract/eval.py`, `tesseract/risk.py` |
| **Resident** | A character who experiences past and future at once | `tesseract/lora_voice.py`, `tesseract/character.py` |
| **Smear** | Video and audio 4D smearing | `tesseract/life.py` |

The heavy training run is still a notebook for a Colab Pro A100. This repo is that model, the four studios, and the Tauri 2 platform around them.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AaronGrace978/4D-Mind/blob/main/notebooks/4d_mind_a100.ipynb)

## Platform

```bash
pip install -r requirements.txt
python3 -m tesseract.demo
```

Open `http://127.0.0.1:8765`. The four studio tabs sit above the solid. Drag to orbit, move **Rotate time into x**, drag the cutting plane, and click a point to read the interior.

```bash
python3 -m tesseract.demo --sculpture doom
python3 -m tesseract.demo --repo .
python3 -m tesseract.demo --npz iris_block.npz
```

The **Hold** box swaps the solid without restarting: paste a git path or a `.npz`, type `linux` / `doom` / `git`, or leave it blank for Iris.

### Sculpt

Famous trees become orbitable solids. Without a clone, the platform holds a characteristic ambient block (deep nesting for Linux, compact scar-heavy C for Doom). With a local checkout, `tesseract.codeblock` rasterizes the real history.

```bash
python3 -m tesseract.codeblock --sculpture linux --out linux_block.npz
python3 -m tesseract.codeblock path/to/DOOM --out doom_block.npz
```

### Scan

A causal risk head ranks spatial patches by whether a fix lands there later. The geometric heuristic needs only numpy: cumulative churn, nearby fix filaments, other authors, definitions, and nesting depth. Each slice may use only that slice and earlier ones.

```bash
python3 -m tesseract.risk scan .
python3 -m tesseract.risk backtest .
python3 -m tesseract.risk train repo_a repo_b --gallery doom --out checkpoints/risk.pt
```

Point `FOURD_MIND_RISK_CKPT` at a trained head to blend it into the Scan studio. This ranks historically fix-prone geometry; it does not produce exploits.

### Resident

You stand at one instant `t`. The resident answers from there, and from every other instant of the block. Three voices share the same facts:

1. **grounded** — sentences built from the block’s measurements (always on)
2. **llm** — any OpenAI-compatible server (`FOURD_MIND_LLM_URL`, `FOURD_MIND_LLM_MODEL`)
3. **lora** — the Qwen LoRA from `tesseract.lora_voice` (`FOURD_MIND_LORA`, `FOURD_MIND_RESIDENT_CKPT`)

### Smear

A clip becomes the same five-channel block a life uses. Frames occupy x and y; t is a Gaussian streak; z lifts bright samples and audio energy off the picture plane. Built-in kinds: `pulse`, `sweep`, `flicker`, `chord`, `iris-echo`. A path to a `.wav` stains the affect channel.

## What gets trained

1. **Pretrain** the 4D net on procedural lives. It reconstructs held-out slices (a future instant and a far side of the body are the same kind of task), probes interior points, and regresses whole-block facts: how long the tube is, where a scar opens and stays, where a second worldline touches.
2. **Fine-tune** that net on [Iris](lives/iris.json). Her childhood room and her deathbed are the same coordinates. Those moments are neighbors along time the way two rooms are neighbors along x.
3. **LoRA-fine-tune Qwen**. The block latent becomes soft-prompt tokens. On a 40 GB A100 the notebook uses Qwen2.5-3B-Instruct. On an 80 GB A100 it steps up to Qwen2.5-7B-Instruct. The utterances it learns are written from the simulator’s own measurements, so the voice is glued to the volume. Character turns place the listener at a random instant so the voice has to speak past and future in the same breath.
4. A **causal twin** of the same architecture, masked so a token cannot see a later time. At a cut through the block, that witness has not met events on the far side of the cut. The resident has.
5. The **risk head** reads that causal pass and ranks patches for a later fix.

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

## A repository as a solid

A git history rasterizes into the same block a life does, with the same five channels, so the viewer and the 4D net take it unchanged.

| Axis | In a repository |
| --- | --- |
| t | commit, oldest to newest along `main`'s first-parent history |
| x | file, sorted by path, so a directory is a neighborhood |
| y | line position, top of the file at the top of the block |
| z | nesting depth (leading indentation) |

| Channel | Iris | Repository |
| --- | --- | --- |
| occupancy | body | a non-blank line of code |
| tissue | the heart | definitions: `def`, `class`, `fn`, `function`, `struct`, ... |
| affect | mood | churn: the share of lines here that changed since the previous slice |
| scar | opens and stays | lines written by a commit whose message says fix, bug, or revert, kept while they survive |
| other | Jun | lines written by anyone but the repository's main author |

A line keeps its author and its fix flag for as long as it survives later edits. A merge's lines take theirs from `git blame`, so a merged PR is credited to whoever wrote it, not whoever pressed merge.

```bash
python3 -m tesseract.demo --repo .                        # view this repo's own history
python3 -m tesseract.codeblock path/to/repo --out repo_block.npz
python3 -m tesseract.codeblock path/to/repo --shape 32 64 64 64 --out repo_a100.npz   # the a100 block size
```

Clicking a point names the commit, the file, the approximate line, and what is there. The `.npz` has the same layout as `iris_block.npz`, so `tesseract.life.load_npz` and the eval functions read it.

## Desktop app (Tauri 2.12)

`desktop/` wraps the platform in a native window. The app starts `python3 -m tesseract.demo` on a free local port, shows the studios once they answer, and stops the server when the window closes.

Needs Rust, Node 18+, Python with `numpy`, and on Linux the WebKitGTK headers (`libwebkit2gtk-4.1-dev libgtk-3-dev librsvg2-dev`; see [Tauri's prerequisites](https://v2.tauri.app/start/prerequisites/)).

```bash
cd desktop
npm install
npm run dev                                   # Iris
FOURD_MIND_REPO=/path/to/repo npm run dev     # a repository's history
npm run build                                 # installers in src-tauri/target/release/bundle/
```

`FOURD_MIND_NPZ` opens an exported block, `FOURD_MIND_PYTHON` picks the interpreter, and `FOURD_MIND_ROOT` points a built app at this checkout if it was moved. A built binary also forwards its arguments, so `4d-mind --repo .` and `4d-mind --sculpture doom` work.

## Tests

```bash
pip install -r requirements.txt
python3 -m pytest
```

The CPU tests use the `tiny` preset. They check Iris’s geometry (the two ends of the tube share a room, the scar persists, tissue sits inside the body), that a causal mask cannot see the future, and that a resident who can see the rest of the block reconstructs a hidden early slice that the witness cannot. The repository tests build a small git history and check that a deleted file leaves the solid, fix lines persist after the fix, and another author's lines show up where that author wrote them. Platform tests hold an ambient Doom sculpture, smear a clip, ask the grounded resident, and hit the studio APIs.

## Layout

| Path | Role |
| --- | --- |
| `tesseract/life.py` | Procedural lives, Iris, and the video/audio smear rasterizer |
| `tesseract/model.py` | 4D patch transformer, slice / point / invariant / risk heads, soft-prompt bridge |
| `tesseract/train.py` | Pretrain, and the causal twin |
| `tesseract/finetune.py` | Adapt the block net onto Iris |
| `tesseract/lora_voice.py` | LoRA stage for the simultaneous character |
| `tesseract/character.py` | Grounded / LLM / LoRA resident |
| `tesseract/eval.py` | Early vs late slice error, interior error, rotation cosine, fix-risk scores |
| `tesseract/risk.py` | Geometric heuristic, causal risk training, scan / backtest CLI |
| `tesseract/codeblock.py` | A git history or ambient sculpture as a block |
| `tesseract/demo.py`, `web/` | Platform viewer |
| `desktop/` | Tauri 2.12 shell around the platform |
| `tesseract/gradio_app.py` | Colab instrument |
| `notebooks/4d_mind_a100.ipynb` | The A100 entry point, written by `notebooks/build_notebook.py` |

`python3 -m tesseract.train --config a100` and `python3 -m tesseract.finetune --ckpt checkpoints/eternalist.pt` are the same functions the notebook calls.
