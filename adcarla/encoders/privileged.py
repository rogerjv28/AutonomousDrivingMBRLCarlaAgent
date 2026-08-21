"""PrivilegedBEVEncoder: CNN sobre la máscara BEV semántica -> embedding (encoder del profesor)."""
import torch.nn as nn
from .base import BEVEncoder

class PrivilegedBEVEncoder(BEVEncoder):
    """Encoder del profesor: reduce la máscara BEV privilegiada [C,H,W] a un embedding [K, embed_dim]."""

    def __init__(self, in_channels: int, embed_dim: int, size: int = 128):
        """CNN de 3 bloques convolucionales + pooling global + proyección lineal.

        Args:
            in_channels: numero de canales BEV de entrada (cfg["bev"]["channels"]).
            embed_dim: dimensión del embedding de salida, la que espera el RSSM del world model
                (cfg["world_model"]["embed_dim"]).
            size: tamaño H=W de la rejilla BEV de entrada (cfg["bev"]["size"]).
        """
        super().__init__()
        self.embed_dim = embed_dim
        # 3 capas convolucionales que reducen el tamaño a la mitad y duplican las dimensiones
        # con la función de activación Sigmoid Linear Unit y una capa de pooling que hace la 
        # media de cada dimension.
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, inputs: dict):
        x = inputs["bev"]   # [K, C, H, W]
        return self.fc(self.conv(x).flatten(1)) # el flatten canvia el tensor de [K, 128, 1, 1] a [K, 128]
