"""Volume and model presets.

`tiny` fits a CPU overfit. `a100` is the Colab Pro network: a 32×64×64×64
block, patch 8, width 768, depth 12.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass
class TrainConfig:
    name: str = "tiny"
    t: int = 8
    d: int = 8
    h: int = 8
    w: int = 8
    patch: int = 4
    in_ch: int = 5
    dim: int = 32
    depth: int = 2
    heads: int = 4
    mlp_ratio: float = 2.0
    n_invariants: int = 10
    n_soft_prompts: int = 8
    lm_hidden: int = 32
    dropout: float = 0.0
    lr: float = 3e-4
    finetune_lr: float = 1e-4
    weight_decay: float = 0.0
    mask_ratio: float = 0.35
    rotation_degrees: float = 25.0
    rotation_every: int = 1
    grad_checkpoint: bool = False
    interior_queries: int = 8
    seed: int = 7

    def __post_init__(self) -> None:
        self.patch = int(self.patch)
        if self.dim % self.heads != 0:
            raise ValueError(f"dim {self.dim} is not divisible by heads {self.heads}")
        for label, size in (("t", self.t), ("d", self.d), ("h", self.h), ("w", self.w)):
            if size % self.patch != 0:
                raise ValueError(f"{label}={size} is not divisible by patch {self.patch}")

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (self.t, self.d, self.h, self.w)

    def to_dict(self) -> dict:
        return asdict(self)


def tiny() -> TrainConfig:
    return TrainConfig()


def a100() -> TrainConfig:
    return TrainConfig(
        name="a100",
        t=32,
        d=64,
        h=64,
        w=64,
        patch=8,
        dim=768,
        depth=12,
        heads=12,
        mlp_ratio=4.0,
        lm_hidden=2048,
        dropout=0.0,
        lr=2e-4,
        finetune_lr=4e-5,
        weight_decay=0.01,
        mask_ratio=0.45,
        rotation_degrees=30.0,
        rotation_every=4,
        grad_checkpoint=True,
        interior_queries=32,
        seed=7,
    )


def config_from_dict(payload: dict) -> TrainConfig:
    known = {item.name for item in fields(TrainConfig)}
    return TrainConfig(**{key: value for key, value in payload.items() if key in known})


def preset(name: str) -> TrainConfig:
    if name == "tiny":
        return tiny()
    if name == "a100":
        return a100()
    raise ValueError(f"unknown preset {name!r}; use 'tiny' or 'a100'")


def a100_runtime(vram_gb: float, ram_gb: float, run: str) -> dict:
    """Batch, cache, and language-model choice for a Colab GPU."""
    if run not in {"session", "long"}:
        raise ValueError("run must be 'session' or 'long'")
    if vram_gb >= 70:
        language_model = "Qwen/Qwen2.5-7B-Instruct"
        batch = 2
        lm_batch = 2
    elif vram_gb >= 36:
        language_model = "Qwen/Qwen2.5-3B-Instruct"
        batch = 2
        lm_batch = 1
    else:
        language_model = "Qwen/Qwen2.5-3B-Instruct"
        batch = 1
        lm_batch = 1
    steps = {
        "session": {"pretrain": 600, "finetune": 200, "causal": 200, "lora": 250},
        "long": {"pretrain": 6000, "finetune": 1000, "causal": 800, "lora": 1200},
    }[run]
    # One a100-resolution life is about 160 MB.
    cache = max(4, min(48, int((ram_gb - 6) / 0.2)))
    return {
        "language_model": language_model,
        "batch": batch,
        "lm_batch": lm_batch,
        "steps": steps,
        "cache": cache,
    }
