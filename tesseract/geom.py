"""Rotations that mix time with a spatial axis.

A 4D resident can turn the block until duration points the way left/right used to.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def rotate_xt(volume: torch.Tensor, degrees: float) -> torch.Tensor:
    """Rotate `volume` in the time–x plane.

    `volume` is `(batch, channel, time, depth, height, width)`.
    `degrees` is the turn of the time axis toward x. Zero returns a copy
    sampled on the same grid, so it stays aligned with `grid_sample`.
    """
    if volume.ndim != 6:
        raise ValueError(f"expected (B, C, T, D, H, W), got {tuple(volume.shape)}")
    batch, channels, time, depth, height, width = volume.shape
    angle = math.radians(float(degrees))
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    t_axis = torch.linspace(-1.0, 1.0, time, device=volume.device, dtype=volume.dtype)
    x_axis = torch.linspace(-1.0, 1.0, width, device=volume.device, dtype=volume.dtype)
    grid_t, grid_x = torch.meshgrid(t_axis, x_axis, indexing="ij")
    # Sample the inverse rotation so the block's contents turn by `degrees`.
    src_t = cos_a * grid_t + sin_a * grid_x
    src_x = -sin_a * grid_t + cos_a * grid_x
    grid = torch.stack([src_x, src_t], dim=-1)
    flat = volume.permute(0, 1, 3, 4, 2, 5).reshape(batch * channels * depth * height, 1, time, width)
    grid = grid.expand(flat.shape[0], time, width, 2)
    sampled = F.grid_sample(flat, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    restored = sampled.view(batch, channels, depth, height, time, width)
    return restored.permute(0, 1, 4, 2, 3, 5).contiguous()
