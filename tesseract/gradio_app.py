"""Gradio instrument for the Colab A100: the solid, the resident, and the witness."""

from __future__ import annotations

import numpy as np

from tesseract.life import load_iris, rasterize
from tesseract.train import load_checkpoint, volumes_to_torch
from tesseract.viewdata import sample_points, slice_image
from tesseract.voice_text import QUERY_TEXT, describe


def _figure_image(draw) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig = draw(plt)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return image


def render_projection(volume: np.ndarray, angle_deg: float) -> np.ndarray:
    points = sample_points(volume, max_points=6000, seed=1)
    if len(points) == 0:
        points = np.zeros((1, 9), dtype=np.float32)

    def draw(plt):
        angle = np.deg2rad(angle_deg)
        x = points[:, 0] - 0.5
        y = points[:, 1] - 0.5
        t = points[:, 3] - 0.5
        xr = x * np.cos(angle) - t * np.sin(angle)
        tr = x * np.sin(angle) + t * np.cos(angle)
        fig = plt.figure(figsize=(6.2, 5.4), facecolor="#07080b")
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
        from pathlib import Path

        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from tesseract.lora_voice import generate_from_latent

        bridge_blob = torch.load(Path(lora_dir) / "bridge.pt", map_location="cpu", weights_only=False)
        tokenizer = AutoTokenizer.from_pretrained(lora_dir)
        base = AutoModelForCausalLM.from_pretrained(
            bridge_blob["base"], torch_dtype=torch.bfloat16, device_map="auto"
        )
        language = PeftModel.from_pretrained(base, lora_dir)
        language.eval()
        resident.set_lm_hidden(int(bridge_blob["lm_hidden"]))
        resident.soft.load_state_dict(bridge_blob["soft"])
        bridge = resident.soft.to(device_t)
        bridge.eval()

        def resident_says(query: str) -> str:
            with torch.no_grad():
                latent = resident.latent_only(volumes_to_torch(volume[None].astype(np.float32), device_t))
            return generate_from_latent(language, tokenizer, bridge, latent[0], query)

    else:

        def resident_says(query: str) -> str:
            del query
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

    with gr.Blocks(title="4D Block Mind") as demo:
        gr.Markdown(
            "# 4D Block Mind\nIris, held as one solid. "
            "Turn time into a direction, cut the block, and ask the resident who already sees both ends."
        )
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
    return demo


def launch(resident_ckpt: str, causal_ckpt: str | None = None, lora_dir: str | None = None, share: bool = False) -> None:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    demo = build_interface(resident_ckpt, causal_ckpt=causal_ckpt, lora_dir=lora_dir, device=device)
    demo.launch(share=share)
