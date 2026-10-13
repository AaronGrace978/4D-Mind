"""Volume and model presets.

`tiny` fits a CPU overfit. `repo` is the fix-risk predictor's CPU-sized net.
`rtx16gb` fits a 16 GB desktop GPU (such as an RTX 5060 Ti) or a free Colab T4.
`a100` is the Colab Pro network: a 32×64×64×64 block, patch 8, width 768, depth 12.
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


def repo() -> TrainConfig:
    """The risk predictor's block: 16 time slices, 8 nesting layers, 32 line rows, 32 file columns.

    Small enough to train on a CPU in minutes.
    """
    return TrainConfig(
        name="repo",
        t=16,
        d=8,
        h=32,
        w=32,
        patch=4,
        dim=96,
        depth=3,
        heads=4,
        mlp_ratio=2.0,
        lr=1e-3,
        finetune_lr=3e-4,
        weight_decay=0.01,
        mask_ratio=0.3,
        seed=7,
    )


def rtx16gb() -> TrainConfig:
    """16 GB desktop GPU (RTX 5060 Ti) or free Colab T4: 16×32×32×32 block, patch 4, 512 tokens."""
    return TrainConfig(
        name="rtx16gb",
        t=16,
        d=32,
        h=32,
        w=32,
        patch=4,
        dim=384,
        depth=6,
        heads=6,
        mlp_ratio=3.0,
        lm_hidden=2048,
        dropout=0.0,
        lr=2.5e-4,
        finetune_lr=6e-5,
        weight_decay=0.01,
        mask_ratio=0.40,
        rotation_degrees=28.0,
        rotation_every=3,
        grad_checkpoint=True,
        interior_queries=24,
        seed=7,
    )


def config_from_dict(payload: dict) -> TrainConfig:
    known = {item.name for item in fields(TrainConfig)}
    return TrainConfig(**{key: value for key, value in payload.items() if key in known})


def preset(name: str) -> TrainConfig:
    if name == "tiny":
        return tiny()
    if name == "repo":
        return repo()
    if name in {"rtx16gb", "16gb", "5060ti", "t4"}:
        return rtx16gb()
    if name == "a100":
        return a100()
    raise ValueError(f"unknown preset {name!r}; use 'tiny', 'repo', 'rtx16gb', or 'a100'")


def a100_runtime(vram_gb: float, ram_gb: float, run: str, family: str = "qwen") -> dict:
    """Batch, cache, config preset, and language-model choice for a GPU."""
    if run not in {"session", "long"}:
        raise ValueError("run must be 'session' or 'long'")
    family = family.strip().lower()
    if vram_gb >= 70:
        language_model = (
            "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
            if family == "deepseek"
            else ("Qwen/Qwen2.5-Coder-7B-Instruct" if family == "qwen-coder" else "Qwen/Qwen2.5-7B-Instruct")
        )
        batch = 2
        lm_batch = 2
        cfg_name = "a100"
    elif vram_gb >= 36:
        language_model = (
            "deepseek-ai/deepseek-coder-6.7b-instruct"
            if family == "deepseek"
            else ("Qwen/Qwen2.5-Coder-3B-Instruct" if family == "qwen-coder" else "Qwen/Qwen2.5-3B-Instruct")
        )
        batch = 2
        lm_batch = 1
        cfg_name = "a100"
    else:
        language_model = (
            "deepseek-ai/deepseek-coder-1.3b-instruct"
            if family == "deepseek"
            else ("Qwen/Qwen2.5-Coder-3B-Instruct" if family == "qwen-coder" else "Qwen/Qwen2.5-3B-Instruct")
        )
        batch = 1
        lm_batch = 1
        cfg_name = "rtx16gb" if vram_gb < 24 else "a100"
    steps = {
        "session": {"pretrain": 600, "finetune": 200, "causal": 200, "lora": 250},
        "long": {"pretrain": 6000, "finetune": 1000, "causal": 800, "lora": 1200},
    }[run]
    # One a100-resolution life is about 160 MB; an rtx16gb life is 10 MB.
    per_life_gb = 0.02 if cfg_name == "rtx16gb" else 0.2
    cache = max(4, min(48, int((ram_gb - 6) / per_life_gb)))
    return {
        "config": cfg_name,
        "block_config": cfg_name,
        "language_model": language_model,
        "batch": batch,
        "lm_batch": lm_batch,
        "steps": steps,
        "cache": cache,
    }
