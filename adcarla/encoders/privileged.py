"""PrivilegedBEVEncoder: CNN sobre la máscara BEV semántica -> rejilla BEV -> embedding (profesor)."""
import torch
import torch.nn as nn
from .base import BEVEncoder, BEVGridToEmbedding

# Canales de la rejilla BEV que expone bev_features(): salida de la última conv del stack.
# Coincide a propósito con el `bev_channels` por defecto de los alumnos.
GRID_CHANNELS = 128
# Factor de reducción del stack convolucional (3 bloques stride 2): con bev.size = 128 deja
# exactamente la rejilla de 16x16. Lo usa build_encoder() para validar el config.
PRIVILEGED_REDUCTION = 8


class PrivilegedBEVEncoder(BEVEncoder):
    """Encoder del profesor: reduce la máscara BEV privilegiada [C,H,W] a un embedding [K, embed_dim]."""

    input_keys = ("bev", "measurements")

    def __init__(self, in_channels: int, embed_dim: int, grid_size: int = 16):
        """CNN de 3 bloques convolucionales -> rejilla BEV -> compresión a embedding.

        Args:
            in_channels: numero de canales BEV de entrada (cfg["bev"]["channels"]).
            embed_dim: dimensión del embedding de salida, la que espera el RSSM del world model
                (cfg["world_model"]["embed_dim"]).
            grid_size: lado de la rejilla BEV de características (cfg["bev"]["grid_size"]), el
                mismo para los tres encoders.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.grid_channels = GRID_CHANNELS

        # 3 capas convolucionales que reducen el tamaño a la mitad y duplican las dimensiones
        # con la función de activación Sigmoid Linear Unit. El pooling final NO colapsa a un
        # vector: fija el lado de la rejilla a grid_size sea cual sea bev.size (con bev.size=128
        # es la identidad, 128/8 = 16).
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(64, GRID_CHANNELS, 4, 2, 1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((grid_size, grid_size)),
        )
        self.to_embedding = BEVGridToEmbedding(GRID_CHANNELS, grid_size, embed_dim)
        self._init_measurement_head(embed_dim)  # el profesor tambien recibe el vector de las medidas del ego

    def bev_features(self, inputs: dict) -> torch.Tensor:
        """Rejilla BEV de características del profesor.

        Args:
            inputs: dict con la clave "bev" -> máscara privilegiada [K, C, H, W].

        Returns:
            [K, GRID_CHANNELS, grid_size, grid_size].
        """
        if "bev" not in inputs:
            raise RuntimeError("PrivilegedBEVEncoder.bev_features() requiere la clave 'bev' en inputs")
        return self.conv(inputs["bev"])
