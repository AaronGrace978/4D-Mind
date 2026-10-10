"""Non-causal 4D transformer.

A life is patched into tokens that all attend to each other. Time is one of
four coordinates on the token, not an order the network is forced to respect.
A causal mask is optional: that pass is the 3D witness, who cannot use a later
slice to fill in an earlier one.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from tesseract.config import TrainConfig


class FourierEncode(nn.Module):
    def __init__(self, n_coords: int, out_dim: int, n_freq: int = 6) -> None:
        super().__init__()
        self.register_buffer("freqs", (2.0 ** torch.arange(n_freq)) * torch.pi, persistent=False)
        self.proj = nn.Linear(n_coords * n_freq * 2, out_dim)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        angles = coords.unsqueeze(-1) * self.freqs.to(dtype=coords.dtype)
        features = torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(-2)
        return self.proj(features)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim {dim} is not divisible by {heads} heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, hidden: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        batch, tokens, channels = hidden.shape
        qkv = self.qkv(hidden).reshape(batch, tokens, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        mixed = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
        mixed = mixed.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(mixed)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        hidden = hidden + self.drop(self.attn(self.norm1(hidden), attn_mask))
        hidden = hidden + self.mlp(self.norm2(hidden))
        return hidden


class BlockOutput:
    def __init__(
        self,
        latent: torch.Tensor,
        tokens: torch.Tensor,
        patch_pred: torch.Tensor,
        patch_target: torch.Tensor,
        invariants: torch.Tensor,
        soft_prompts: torch.Tensor,
        interior: torch.Tensor | None,
    ) -> None:
        self.latent = latent
        self.tokens = tokens
        self.patch_pred = patch_pred
        self.patch_target = patch_target
        self.invariants = invariants
        self.soft_prompts = soft_prompts
        self.interior = interior


class Eternalist(nn.Module):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch = (cfg.patch, cfg.patch, cfg.patch, cfg.patch)
        pt, pz, py, px = self.patch
        self.n_t = cfg.t // pt
        self.n_z = cfg.d // pz
        self.n_y = cfg.h // py
        self.n_x = cfg.w // px
        self.n_tokens = self.n_t * self.n_z * self.n_y * self.n_x
        self.patch_dim = cfg.in_ch * pt * pz * py * px

        self.embed = nn.Linear(self.patch_dim, cfg.dim)
        self.pos = FourierEncode(4, cfg.dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList(
            Block(cfg.dim, cfg.heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.depth)
        )
        self.norm = nn.LayerNorm(cfg.dim)
        self.patch_head = nn.Linear(cfg.dim, self.patch_dim)
        self.invariant_head = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim),
            nn.GELU(),
            nn.Linear(cfg.dim, cfg.n_invariants),
        )
        self.point_pos = FourierEncode(4, cfg.dim)
        self.point_head = nn.Sequential(
            nn.Linear(cfg.dim * 2, cfg.dim),
            nn.GELU(),
            nn.Linear(cfg.dim, cfg.in_ch),
        )
        self.soft = nn.Linear(cfg.dim, cfg.n_soft_prompts * cfg.lm_hidden)

        patch_index, coords = _token_tables(self.n_t, self.n_z, self.n_y, self.n_x, self.patch, cfg.shape)
        self.register_buffer("patch_index", patch_index, persistent=True)
        self.register_buffer("coords", coords, persistent=True)

    def set_lm_hidden(self, hidden: int) -> None:
        hidden = int(hidden)
        if hidden == self.cfg.lm_hidden:
            return
        device = self.soft.weight.device
        dtype = self.soft.weight.dtype
        self.soft = nn.Linear(self.cfg.dim, self.cfg.n_soft_prompts * hidden).to(device=device, dtype=dtype)
        self.cfg.lm_hidden = hidden

    def perception_parameters(self):
        for name, param in self.named_parameters():
            if name.startswith("soft."):
                continue
            yield param

    def soft_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        batch = latent.shape[0]
        return self.soft(latent).view(batch, self.cfg.n_soft_prompts, self.cfg.lm_hidden)

    def patchify(self, volume: torch.Tensor) -> torch.Tensor:
        batch, channels, time, depth, height, width = volume.shape
        pt, pz, py, px = self.patch
        if (time, depth, height, width) != self.cfg.shape:
            raise ValueError(f"volume spatial shape {(time, depth, height, width)} != {self.cfg.shape}")
        if channels != self.cfg.in_ch:
            raise ValueError(f"expected {self.cfg.in_ch} channels, got {channels}")
        view = volume.reshape(batch, channels, self.n_t, pt, self.n_z, pz, self.n_y, py, self.n_x, px)
        view = view.permute(0, 2, 4, 6, 8, 1, 3, 5, 7, 9).contiguous()
        return view.reshape(batch, self.n_tokens, self.patch_dim)

    def unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        batch = tokens.shape[0]
        pt, pz, py, px = self.patch
        channels = self.cfg.in_ch
        view = tokens.view(batch, self.n_t, self.n_z, self.n_y, self.n_x, channels, pt, pz, py, px)
        view = view.permute(0, 5, 1, 6, 2, 7, 3, 8, 4, 9).contiguous()
        return view.view(batch, channels, self.cfg.t, self.cfg.d, self.cfg.h, self.cfg.w)

    def slice_mask(self, batch: int, axis: str, index: int, device: torch.device) -> torch.Tensor:
        axis_id = {"t": 0, "z": 1, "y": 2, "x": 3}[axis]
        hidden = self.patch_index[:, axis_id] == int(index)
        mask = hidden.view(1, -1).expand(batch, -1).clone()
        if bool(hidden.all()):
            mask[:, 0] = False
        return mask.to(device)

    def attention_mask(self, causal: bool, device: torch.device) -> torch.Tensor | None:
        """`None` for the resident, so SDPA can use its fused kernels.

        The witness gets a boolean mask: True where a query may look, which is
        any key at the same time or earlier.
        """
        if not causal:
            return None
        time_index = self.t_index.to(device)
        return time_index.unsqueeze(0) <= time_index.unsqueeze(1)

    @property
    def t_index(self) -> torch.Tensor:
        return self.patch_index[:, 0]

    def forward(
        self,
        volume: torch.Tensor,
        patch_mask: torch.Tensor | None = None,
        causal: bool = False,
        points: torch.Tensor | None = None,
        until_bin: int | torch.Tensor | None = None,
    ) -> BlockOutput:
        target = self.patchify(volume)
        hidden = self.embed(target) + self.pos(self.coords)
        if patch_mask is not None:
            token = self.mask_token.to(dtype=hidden.dtype, device=hidden.device).expand_as(hidden)
            hidden = torch.where(patch_mask.unsqueeze(-1), token, hidden)
        attn_mask = self.attention_mask(causal, hidden.device)
        for block in self.blocks:
            if self.training and self.cfg.grad_checkpoint:
                hidden = checkpoint(block, hidden, attn_mask, use_reentrant=False)
            else:
                hidden = block(hidden, attn_mask)
        hidden = self.norm(hidden)
        latent = self._pool(hidden, patch_mask, causal, until_bin)
        interior = None
        if points is not None:
            query = self.point_pos(points)
            pooled = latent.unsqueeze(1).expand(-1, query.shape[1], -1)
            interior = self.point_head(torch.cat([pooled, query], dim=-1))
        return BlockOutput(
            latent=latent,
            tokens=hidden,
            patch_pred=self.patch_head(hidden),
            patch_target=target,
            invariants=self.invariant_head(latent),
            soft_prompts=self.soft_from_latent(latent),
            interior=interior,
        )

    def latent_only(
        self,
        volume: torch.Tensor,
        causal: bool = False,
        until_bin: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward(volume, causal=causal, until_bin=until_bin).latent

    def _pool(
        self,
        hidden: torch.Tensor,
        patch_mask: torch.Tensor | None,
        causal: bool,
        until_bin: int | torch.Tensor | None,
    ) -> torch.Tensor:
        batch, tokens, _dim = hidden.shape
        time_index = self.t_index
        if causal and until_bin is None:
            until_bin = int(time_index.max().item())
        if until_bin is None:
            use = torch.ones(batch, tokens, dtype=torch.bool, device=hidden.device)
        elif isinstance(until_bin, int):
            use = (time_index <= until_bin).view(1, tokens).expand(batch, tokens)
        else:
            use = time_index.view(1, tokens) <= until_bin.to(hidden.device).view(batch, 1)
        if patch_mask is not None:
            use = use & ~patch_mask
        weight = use.float()
        denom = weight.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (hidden * weight.unsqueeze(-1)).sum(dim=1) / denom


def _token_tables(
    n_t: int,
    n_z: int,
    n_y: int,
    n_x: int,
    patch: tuple[int, int, int, int],
    shape: tuple[int, int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    time, depth, height, width = torch.meshgrid(
        torch.arange(n_t),
        torch.arange(n_z),
        torch.arange(n_y),
        torch.arange(n_x),
        indexing="ij",
    )
    index = torch.stack([time, depth, height, width], dim=-1).reshape(-1, 4)
    centers = []
    for axis, count, patch_size, size in zip(range(4), (n_t, n_z, n_y, n_x), patch, shape):
        center = index[:, axis].float() * patch_size + (patch_size - 1) / 2
        centers.append(center / max(size - 1, 1))
    coords = torch.stack(centers, dim=-1)
    return index.long(), coords
