"""Gradio instrument for the Colab A100: the solid, the resident, and the witness."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tesseract.codeblock import ambient_sculpture, rasterize_repo
from tesseract.coder import fourd_code_lens, grounded_code_review
from tesseract.life import load_iris, rasterize
from tesseract.risk import geometric_risk, hotspot_table
from tesseract.train import load_checkpoint, volumes_to_torch
from tesseract.viewdata import sample_points, slice_image
from tesseract.voice_text import QUERY_TEXT, describe


def _figure_image(draw) -> np.ndarray:
    # A bare Agg canvas, so the notebook's inline pyplot backend is left alone.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = draw(Figure)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[:, :, :3].copy()


def render_projection(volume: np.ndarray, angle_deg: float) -> np.ndarray:
    points = sample_points(volume, max_points=6000, seed=1)
    if len(points) == 0:
        points = np.zeros((1, 9), dtype=np.float32)

    def draw(Figure):
        angle = np.deg2rad(angle_deg)
        x = points[:, 0] - 0.5
        y = points[:, 1] - 0.5
        t = points[:, 3] - 0.5
        xr = x * np.cos(angle) - t * np.sin(angle)
        tr = x * np.sin(angle) + t * np.cos(angle)
        fig = Figure(figsize=(6.2, 5.4), facecolor="#07080b")
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#07080b")
        scar = points[:, 7] > 0.45
        other = (points[:, 8] > 0.45) & ~scar
        tissue = (points[:, 5] > 0.45) & ~scar & ~other
        body = ~(scar | other | tissue)
        ax.scatter(xr[body], points[body, 2] - 0.5, tr[body], s=3, c="#8ef6d2", alpha=0.35, linewidths=0)
        ax.scatter(xr[tissue], points[tissue, 2] - 0.5, tr[tissue], s=8, c="#ff5d8f", alpha=0.9, linewidths=0)
        ax.scatter(xr[other], points[other, 2] - 0.5, tr[other], s=6, c="#79d2ff", alpha=0.7, linewidths=0)
        ax.scatter(xr[scar], points[scar, 2] - 0.5, tr[scar], s=10, c="#ffb45c", alpha=0.95, linewidths=0)
        ax.set_xlabel("x turned with t")
        ax.set_ylabel("z")
        ax.set_zlabel("t turned with x")
        ax.tick_params(colors="#9aa6b2")
        ax.xaxis.label.set_color("#9aa6b2")
        ax.yaxis.label.set_color("#9aa6b2")
        ax.zaxis.label.set_color("#9aa6b2")
        return fig

    return _figure_image(draw)


def render_slice(volume: np.ndarray, axis: str, value: float) -> np.ndarray:
    return slice_image(volume, axis, value)


def render_risk_map(volume: np.ndarray, meta: dict | None = None) -> np.ndarray:
    """Render commit-time (t) vs file-space (x) causal fix-risk and churn heatmap."""
    del meta
    field = geometric_risk(volume)
    # Collapse z and y to show (t, x) spatiotemporal risk surface.
    tx_map = field.mean(axis=(1, 2))

    def draw(Figure):
        fig = Figure(figsize=(6.2, 5.4), facecolor="#07080b")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#07080b")
        ax.imshow(tx_map, aspect="auto", cmap="magma", origin="lower", vmin=0.0, vmax=1.0)
        ax.set_xlabel("x (file columns across codebase)")
        ax.set_ylabel("t (commit slices: 0=oldest, top=newest)")
        ax.set_title("4D Causal Fix-Risk Field (t × x)", color="#e2e8f0", fontsize=10)
        ax.tick_params(colors="#9aa6b2")
        ax.xaxis.label.set_color("#9aa6b2")
        ax.yaxis.label.set_color("#9aa6b2")
        return fig

    return _figure_image(draw)


def _predict_text(model, volume: np.ndarray, meta: dict, device, causal: bool, until_bin: int | None) -> str:
    import torch

    model.eval()
    tensor = volumes_to_torch(volume[None].astype(np.float32), device)
    with torch.no_grad():
        output = model(tensor, causal=causal, until_bin=until_bin)
    pred = output.invariants[0].float().cpu().numpy()
    spoken = dict(meta)
    spoken["invariants"] = pred.tolist()
    return describe(spoken)


def build_interface(resident_ckpt: str, causal_ckpt: str | None = None, lora_dir: str | None = None, device: str = "cuda"):
    import torch
    import gradio as gr

    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    resident, cfg, _blob = load_checkpoint(resident_ckpt, device_t)
    resident.eval()
    causal = None
    if causal_ckpt:
        causal, _ccfg, _cblob = load_checkpoint(causal_ckpt, device_t)
        causal.eval()
    volume, _inv, meta = rasterize(load_iris(), cfg.shape)
    language = tokenizer = bridge = None
    if lora_dir:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from tesseract.lora_voice import dtype_kwargs, generate_from_latent

        bridge_blob = torch.load(Path(lora_dir) / "bridge.pt", map_location="cpu", weights_only=False)
        tokenizer = AutoTokenizer.from_pretrained(lora_dir)
        base = AutoModelForCausalLM.from_pretrained(
            bridge_blob["base"], device_map="auto", **dtype_kwargs(torch.bfloat16)
        )
        language = PeftModel.from_pretrained(base, lora_dir)
        language.eval()
        resident.set_lm_hidden(int(bridge_blob["lm_hidden"]))
        resident.soft.load_state_dict(bridge_blob["soft"])
        bridge = resident.soft.to(device_t)
        bridge.eval()

        def resident_says(query: str, solid: np.ndarray | None = None) -> str:
            target = volume if solid is None else solid
            with torch.no_grad():
                latent = resident.latent_only(volumes_to_torch(target[None].astype(np.float32), device_t))
            return generate_from_latent(language, tokenizer, bridge, latent[0], query)

    else:

        def resident_says(query: str, solid: np.ndarray | None = None) -> str:
            del query, solid
            return describe(meta)

    def look(angle, axis, cut, query):
        projection = render_projection(volume, float(angle))
        plane = render_slice(volume, axis, float(cut))
        resident_text = resident_says(query or QUERY_TEXT["describe"])
        if causal is None:
            witness_text = "The causal twin checkpoint is not loaded."
        else:
            until = min(causal.n_t - 1, max(0, int(float(cut) * causal.n_t)))
            witness_text = _predict_text(causal, volume, meta, device_t, causal=True, until_bin=until)
        return projection, plane, resident_text, witness_text

    default_repo = "/content/4D-Mind" if Path("/content/4D-Mind/.git").exists() else str(Path.cwd())

    def code_4d(target_spec, angle, code_query):
        spec = (target_spec or default_repo).strip()
        repo_path: Path | None = None
        if spec in {"linux", "doom", "git", "cpython"}:
            c_vol, _c_inv, c_meta = ambient_sculpture(spec, cfg.shape, seed=cfg.seed)
        else:
            candidate = Path(spec).expanduser()
            if (candidate / ".git").exists():
                repo_path = candidate.resolve()
                c_vol, _c_inv, c_meta = rasterize_repo(repo_path, cfg.shape)
            else:
                c_vol, _c_inv, c_meta = ambient_sculpture("linux", cfg.shape, seed=cfg.seed)
        field = geometric_risk(c_vol)
        hotspots = hotspot_table(c_vol, c_meta, field)
        lens = fourd_code_lens(c_vol, c_meta, hotspots=hotspots)
        proj = render_projection(c_vol, float(angle))
        risk_img = render_risk_map(c_vol, c_meta)
        prompt = (code_query or "Analyze this codebase from the 4D lens and propose fixes for the top hotspots.").strip()
        grounded_text, _ = grounded_code_review(c_meta, repo_path, prompt, lens=lens, hotspots=hotspots, focus=None)
        if lora_dir is not None:
            lora_reply = resident_says(prompt, solid=c_vol)
            coder_text = f"### 4D Soft-Prompt LoRA Voice\n{lora_reply}\n\n---\n\n{grounded_text}"
        else:
            coder_text = grounded_text
        return proj, risk_img, str(lens.get("summary", "")), coder_text

    with gr.Blocks(title="4D Block Mind") as demo:
        gr.Markdown(
            "# 4D Block Mind\nHold a life or a codebase as one 4D solid `(t, z, y, x)`. "
            "Turn time into a spatial direction, cut the block, and code from a 4D lens."
        )
        with gr.Tabs():
            with gr.Tab("Resident & Witness (Iris)"):
                with gr.Row():
                    angle = gr.Slider(-180, 180, value=28, step=1, label="Rotate time into x (degrees)")
                    axis = gr.Radio(["t", "x", "y", "z"], value="t", label="Cutting plane")
                    cut = gr.Slider(0, 1, value=0.55, step=0.01, label="Plane position")
                query = gr.Textbox(value=QUERY_TEXT["describe"], label="Ask the resident")
                button = gr.Button("Look")
                with gr.Row():
                    projection = gr.Image(label="Worldtube")
                    plane = gr.Image(label="Cut")
                resident_box = gr.Textbox(label="Block resident", lines=6)
                witness_box = gr.Textbox(label="3D witness at the cut", lines=6)
                button.click(look, [angle, axis, cut, query], [projection, plane, resident_box, witness_box])
                demo.load(look, [angle, axis, cut, query], [projection, plane, resident_box, witness_box])
            with gr.Tab("4D Code Lens & Coder"):
                gr.Markdown(
                    "### 4D Spatiotemporal Code Lens\n"
                    "Rasterize any Git checkout (or ambient sculpture: `linux`, `doom`, `git`, `cpython`) "
                    "into `(t=commit, z=indentation, y=line, x=file)` and inspect its co-churn worldlines, "
                    "temporal fracture slices, and causal future-fix hotspots."
                )
                with gr.Row():
                    repo_input = gr.Textbox(
                        value=default_repo,
                        label="Git repository path or sculpture id (linux, doom, git, cpython)",
                    )
                    code_angle = gr.Slider(-180, 180, value=24, step=1, label="Rotate commit-time t into file-space x")
                code_query = gr.Textbox(
                    value="Analyze this repository from the 4D lens: which files co-mutate along t and where should we refactor?",
                    label="Ask the 4D Coder",
                )
                code_btn = gr.Button("Scan & Code from 4D Lens")
                with gr.Row():
                    code_proj = gr.Image(label="4D Code Worldtube")
                    code_risk = gr.Image(label="Causal Fix-Risk Field (t × x)")
                lens_box = gr.Textbox(label="4D Spatiotemporal Code Lens (worldlines, co-churn coupling, fractures)", lines=8)
                coder_box = gr.Textbox(label="4D Coder Output (LoRA Soft-Prompt + Grounded 4D Audit)", lines=12)
                code_btn.click(
                    code_4d,
                    [repo_input, code_angle, code_query],
                    [code_proj, code_risk, lens_box, coder_box],
                )
    return demo


def public_share(share: bool | None, *, in_colab: bool) -> bool:
    """On Colab, always publish a gradio.live URL.

    Colab rewrites Gradio's localhost link into a ``*.prod.colab.dev`` address, and that address returns 404.
    """
    if in_colab:
        return True
    return False if share is None else share


def launch(resident_ckpt: str, causal_ckpt: str | None = None, lora_dir: str | None = None, share: bool | None = None) -> None:
    import os

    import torch

    in_colab = "COLAB_RELEASE_TAG" in os.environ
    share = public_share(share, in_colab=in_colab)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    demo = build_interface(resident_ckpt, causal_ckpt=causal_ckpt, lora_dir=lora_dir, device=device)
    if in_colab:
        print(
            "Open the https://....gradio.live link printed below. "
            "The *.prod.colab.dev address returns 404."
        )
    demo.launch(share=share)
