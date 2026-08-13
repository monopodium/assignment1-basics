

import math
from collections.abc import Iterable

import torch
from torch import nn
from typing import BinaryIO, IO
import os

def get_lr_cosine_schedule(
    t: int,
    alpha_max: float,
    alpha_min: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if t < warmup_iters:
        return (t / warmup_iters) * alpha_max

    if t <= cosine_cycle_iters:
        progress = (
            (t - warmup_iters)
            / (cosine_cycle_iters - warmup_iters)
        )

        cosine_factor = 0.5 * (
            1 + math.cos(math.pi * progress)
        )

        return (
            alpha_min
            + cosine_factor * (alpha_max - alpha_min)
        )

    return alpha_min




@torch.no_grad()
def gradient_clipping(
    parameters: Iterable[nn.Parameter],
    max_l2_norm: float,
) -> None:
    parameters = list(parameters)

    total_squared_norm = None

    for parameter in parameters:
        if parameter.grad is None:
            continue

        squared_norm = (
            parameter.grad.detach()
            .float()
            .pow(2)
            .sum()
        )

        if total_squared_norm is None:
            total_squared_norm = squared_norm
        else:
            total_squared_norm += squared_norm

    if total_squared_norm is None:
        return

    total_norm = total_squared_norm.sqrt()

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)

        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
                
                
import numpy as np
import torch


def get_batch(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(
        low=0,
        high=len(x) - context_length,
        size=(batch_size,),
    ).numpy()

    offsets = np.arange(context_length)

    input_indices = starts[:, None] + offsets[None, :]
    target_indices = input_indices + 1

    input_array = x[input_indices]
    target_array = x[target_indices]

    inputs = torch.tensor(
        input_array,
        dtype=torch.long,
        device=device,
    )

    targets = torch.tensor(
        target_array,
        dtype=torch.long,
        device=device,
    )

    return inputs, targets

CheckpointSource = str | os.PathLike | BinaryIO | IO[bytes]


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: CheckpointSource,
) -> None:
    """Save model, optimizer, and training iteration to a checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }

    torch.save(checkpoint, out)


def load_checkpoint(
    src: CheckpointSource,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Restore model and optimizer states and return the saved iteration."""
    checkpoint = torch.load(src)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["iteration"]