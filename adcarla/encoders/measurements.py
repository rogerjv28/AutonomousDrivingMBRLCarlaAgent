"""MeasurementEncoder: MLP sobre las medidas del ego.

Vector construido por `CarlaEnv._build_obs()` a partir de `RouteTracker`.
"""
import torch
import torch.nn as nn

from adcarla.carla_env.route_tracker import NUM_COMMANDS

# 1 (velocidad) + 2 (target point) + onehot del comando: derivado de NUM_COMMANDS para que
# no puedan desincronizarse si algun dia cambia el numero de NavigationCommand.
MEASUREMENTS_DIM = 3 + NUM_COMMANDS


class MeasurementEncoder(nn.Module):
    """MLP de 2 capas: MEASUREMENTS_DIM medidas del ego -> hidden_dim -> hidden_dim (SiLU)."""

    def __init__(self, hidden_dim: int = 64):
        """
        Args:
            hidden_dim: dimensión de las dos capas ocultas (64 por defecto).
        """
        super().__init__()
        self.out_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(MEASUREMENTS_DIM, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )

    def forward(self, measurements: torch.Tensor) -> torch.Tensor:
        """
        Args:
            measurements: [K, MEASUREMENTS_DIM].

        Returns:
            [K, hidden_dim].
        """
        return self.net(measurements)
