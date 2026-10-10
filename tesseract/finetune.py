"""Fine-tune a pretrained block model so it resides in Iris."""

from __future__ import annotations

import argparse

from tesseract.train import load_checkpoint, make_cache, pick_device, run_training


def finetune_iris(
    ckpt: str,
    out_path: str,
    steps: int,
    device,
    cache: int = 8,
    seed: int = 7,
    batch_size: int = 2,
    log_every: int = 20,
):
    model, cfg, _blob = load_checkpoint(ckpt, device)
    volumes, invariants, _metas = make_cache(cfg, cache, seed, mode="iris")
    run_training(
        model,
        cfg,
        volumes,
        invariants,
        steps=steps,
        lr=cfg.finetune_lr,
        device=device,
        causal=False,
        kind="iris",
        out_path=out_path,
        batch_size=batch_size,
        seed=seed,
        log_every=log_every,
    )
    return model, cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the block model on Iris")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="checkpoints/iris_resident.pt")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--cache", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    device = pick_device(args.device)
    finetune_iris(
        args.ckpt,
        args.out,
        steps=args.steps,
        device=device,
        cache=args.cache,
        seed=args.seed,
        batch_size=args.batch,
    )
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
