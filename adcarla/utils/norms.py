"""Módulos de normalización compatibles con AMP bfloat16.

Bajo torch.autocast en bfloat16, nn.RMSNorm de PyTorch pierde su kernel
fusionado cuando entrada y peso tienen dtypes distintos, esta subclase
lo restaura casteando el peso al dtype de la entrada.

Uso:

    from adcarla.utils.norms import RMSNorm   # en lugar de nn.RMSNorm
    self.norm = RMSNorm(hidden_dim)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.RMSNorm):
    """``nn.RMSNorm`` con el peso casteado al dtype de la entrada."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.nn.functional.rms_norm(
            x,
            self.normalized_shape,
            self.weight.to(x.dtype),
            self.eps,
        )
