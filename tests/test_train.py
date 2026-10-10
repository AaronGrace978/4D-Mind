import torch

from tesseract.config import tiny
from tesseract.eval import evaluate
from tesseract.life import load_iris, rasterize
from tesseract.model import Eternalist
from tesseract.train import load_checkpoint, make_cache, rotation_loss, run_training, save_checkpoint


def _iris_batch(cfg, copies=4):
    volume, inv, _meta = rasterize(load_iris(), cfg.shape)
    return [volume] * copies, [inv] * copies, volume, inv


def test_loss_falls_and_checkpoint_roundtrip(tmp_path):
    cfg = tiny()
    torch.manual_seed(0)
    model = Eternalist(cfg)
    volumes, invariants, _volume, _inv = _iris_batch(cfg)
    history = run_training(
        model,
        cfg,
        volumes,
        invariants,
        steps=25,
        lr=4e-3,
        device="cpu",
        causal=False,
        rotation_every=5,
        log_every=0,
        batch_size=2,
        seed=0,
    )
    assert history[-1] < history[0] * 0.85
    path = tmp_path / "eternalist.pt"
    save_checkpoint(path, model, 25, "eternalist")
    loaded, loaded_cfg, blob = load_checkpoint(path, "cpu")
    assert blob["kind"] == "eternalist"
    assert loaded_cfg.shape == cfg.shape
    left = model.latent_only(torch.zeros(1, cfg.in_ch, *cfg.shape))
    right = loaded.latent_only(torch.zeros(1, cfg.in_ch, *cfg.shape))
    assert torch.allclose(left, right, atol=1e-5)


def test_rotation_loss_reaches_the_encoder():
    torch.manual_seed(1)
    cfg = tiny()
    model = Eternalist(cfg)
    volume, _inv, _meta = rasterize(load_iris(), cfg.shape)
    from tesseract.train import volumes_to_torch

    tensor = volumes_to_torch(volume[None], "cpu")
    loss = rotation_loss(model, tensor, cfg.rotation_degrees)
    loss.backward()
    assert model.embed.weight.grad is not None
    assert float(model.embed.weight.grad.abs().sum()) > 0
    assert loss.ndim == 0


def test_eternalist_reads_the_early_slice_from_the_rest_of_the_block():
    """Two solids differ only by a constant. The early time slice is hidden.

    The resident can see the later slices, so it can say which solid is which.
    The causal witness, stopped at that early slice, sees only the mask token
    and predicts both solids the same.
    """
    cfg = tiny()
    volume = torch.zeros(2, cfg.in_ch, *cfg.shape)
    volume[0] = 0.15
    volume[1] = 0.85

    def fit(causal: bool) -> float:
        torch.manual_seed(0)
        model = Eternalist(cfg)
        mask = model.slice_mask(2, "t", 0, torch.device("cpu"))
        opt = torch.optim.AdamW(list(model.perception_parameters()), lr=5e-3)
        last = 0.0
        for _ in range(50):
            output = model(volume, patch_mask=mask, causal=causal)
            per = (output.patch_pred - output.patch_target).pow(2).mean(dim=-1)
            loss = (per * mask.float()).sum() / mask.float().sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last = float(loss.detach())
        return last

    assert fit(False) < fit(True) * 0.5


def test_finetune_on_iris_writes_a_resident(tmp_path):
    cfg = tiny()
    torch.manual_seed(0)
    model = Eternalist(cfg)
    volumes, invariants, _volume, _inv = _iris_batch(cfg, copies=2)
    pre = tmp_path / "eternalist.pt"
    run_training(
        model,
        cfg,
        volumes,
        invariants,
        steps=2,
        lr=cfg.lr,
        device="cpu",
        rotation_every=0,
        log_every=0,
        out_path=pre,
        batch_size=2,
    )
    from tesseract.finetune import finetune_iris

    finetune_iris(str(pre), str(tmp_path / "iris_resident.pt"), steps=2, device="cpu", cache=2, batch_size=2, log_every=0)
    _model, _cfg, blob = load_checkpoint(tmp_path / "iris_resident.pt", "cpu")
    assert blob["kind"] == "iris"


def test_evaluate_reports_the_block_scores():
    cfg = tiny()
    torch.manual_seed(3)
    model = Eternalist(cfg)
    volumes, invariants, _metas = make_cache(cfg, 2, seed=3, mode="procedural")
    metrics = evaluate(model, volumes, invariants, "cpu", causal=False)
    for key in ("early_mae", "late_mae", "gap_early_minus_late", "interior_mae", "invariant_mae", "rotation_cosine", "x_slice_mae"):
        assert key in metrics
        assert metrics[key] == metrics[key]
    from tesseract.geom import rotate_xt
    from tesseract.train import volumes_to_torch

    block = volumes_to_torch(volumes[0][None], "cpu")
    cosine = torch.nn.functional.cosine_similarity(
        model.latent_only(block),
        model.latent_only(rotate_xt(block, 0)),
        dim=-1,
    )
    assert float(cosine.detach()) > 0.99
