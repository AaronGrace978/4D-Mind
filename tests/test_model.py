import numpy as np
import torch

from tesseract.config import a100, a100_runtime, preset, rtx16gb, tiny
from tesseract.geom import rotate_xt
from tesseract.lora_voice import attach_soft_prompts, code_examples, labels_from_prompt_lens
from tesseract.model import Eternalist


def test_a100_preset_token_count_and_runtime():
    cfg = a100()
    tokens = (cfg.t // cfg.patch) * (cfg.d // cfg.patch) * (cfg.h // cfg.patch) * (cfg.w // cfg.patch)
    assert tokens == 2048
    runtime = a100_runtime(40, 80, "session")
    assert "Qwen2.5-3B" in runtime["language_model"]
    assert runtime["steps"]["pretrain"] < a100_runtime(80, 80, "long")["steps"]["pretrain"]
    big = a100_runtime(80, 80, "session")
    assert "7B" in big["language_model"]

    # 16 GB RTX 5060 Ti / Colab T4 runtime and DeepSeek-Coder family
    cfg_16 = rtx16gb()
    tokens_16 = (cfg_16.t // cfg_16.patch) * (cfg_16.d // cfg_16.patch) * (cfg_16.h // cfg_16.patch) * (cfg_16.w // cfg_16.patch)
    assert tokens_16 == 2048
    assert cfg_16.dim == 384 and cfg_16.depth == 6
    assert preset("5060ti").shape == cfg_16.shape
    rt_16_ds = a100_runtime(16, 32, "session", family="deepseek")
    assert rt_16_ds["block_config"] == "rtx16gb"
    assert "deepseek-coder" in rt_16_ds["language_model"]
    rt_40_ds = a100_runtime(40, 80, "session", family="deepseek")
    assert "deepseek-coder-6.7b" in rt_40_ds["language_model"]


def test_code_examples_build_4d_repo_and_sculpture_targets():
    cfg = tiny()
    rng = np.random.default_rng(7)
    volumes, metas, rows = code_examples(cfg.shape, rng, n_sculptures=2, include_self_repo=True)
    assert len(volumes) >= 2
    assert volumes[0].shape == cfg.shape + (5,)
    assert metas[0]["kind"] in {"repo", "sculpture"}
    assert len(rows) >= 6
    assert any("4D" in answer or "worldline" in answer.lower() or "slice" in answer.lower() for _idx, _sys, _q, answer in rows)


def test_patchify_roundtrip_and_soft_prompt_shape():
    torch.manual_seed(0)
    cfg = tiny()
    model = Eternalist(cfg)
    volume = torch.rand(2, cfg.in_ch, *cfg.shape)
    tokens = model.patchify(volume)
    assert tokens.shape == (2, model.n_tokens, model.patch_dim)
    restored = model.unpatchify(tokens)
    assert torch.allclose(restored, volume)
    output = model(volume, points=torch.rand(2, 4, 4))
    assert output.soft_prompts.shape == (2, cfg.n_soft_prompts, cfg.lm_hidden)
    assert output.interior.shape == (2, 4, cfg.in_ch)
    assert output.invariants.shape == (2, cfg.n_invariants)
    assert output.risk.shape == (2, model.n_tokens)


def test_zero_rotation_is_the_same_block():
    torch.manual_seed(1)
    volume = torch.randn(1, 2, 8, 4, 4, 8)
    assert torch.allclose(rotate_xt(volume, 0), volume, atol=1e-4)
    assert not torch.allclose(rotate_xt(volume, 35), volume, atol=1e-3)


def test_causal_early_tokens_ignore_the_future():
    torch.manual_seed(2)
    cfg = tiny()
    model = Eternalist(cfg).eval()
    volume = torch.randn(2, cfg.in_ch, *cfg.shape)
    original = model(volume, causal=True)
    altered = volume.clone()
    altered[:, :, cfg.t // 2 :] = 0
    shifted = model(altered, causal=True)
    early = model.t_index == 0
    assert torch.allclose(original.patch_pred[:, early], shifted.patch_pred[:, early], atol=1e-5)
    open_view = model(volume, causal=False)
    open_shifted = model(altered, causal=False)
    assert not torch.allclose(open_view.patch_pred[:, early], open_shifted.patch_pred[:, early], atol=1e-4)


def test_soft_prompt_labels_ignore_the_prefix():
    soft = torch.randn(2, 4, 8)
    token_embeds = torch.randn(2, 6, 8)
    attention = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]])
    ids = torch.tensor([[3, 4, 5, 6, 7, 0], [3, 4, 5, 8, 0, 0]])
    labels = labels_from_prompt_lens(ids, [2, 3], attention)
    inputs, full_attn, full_labels = attach_soft_prompts(soft, token_embeds, attention, labels)
    assert inputs.shape == (2, 10, 8)
    assert torch.all(full_labels[:, :4] == -100)
    assert torch.all(full_labels[0, 4:6] == -100)
    assert torch.equal(full_labels[0, 6:9], ids[0, 2:5])
    assert full_attn[0, 9] == 0
