"""FusionEncoder: BEVFormer (cámara, misma arquitectura que la rama visión) + LiDARBranch + ConvFuser
-> embedding. NO comparte pesos con la rama visión (cada rama entrena su propio encoder)."""
import torch.nn as nn
from ..base import BEVEncoder
from ..bevformer.bevformer import BEVFormerEncoder
from .lidar_branch import LiDARBranch
from .fuser import ConvFuser

class FusionEncoder(BEVEncoder):
    """Encoder de la rama de fusión (cámara + LiDAR): dos ramas BEV independientes -> ConvFuser -> embedding.

    Combina un encoder BEVFormer (cámara) y la LiDARBranch (LiDAR rasterizado),
    ambos con la misma resolución de rejilla BEV, fusiona sus mapas BEV con
    ConvFuser y colapsa el resultado a un embedding final mediante pooling
    global + una capa lineal. No comparte pesos con la rama vision-only.
    """

    def __init__(self, embed_dim: int, bev_channels: int = 128, grid_size: int = 16):
        """Crea las dos ramas (cámara, LiDAR), el fusor y la proyección final.

        Args:
            embed_dim: dimensión del embedding de salida que consume el resto
                del modelo (world model).
            bev_channels: canales internos de los mapas BEV de cámara y LiDAR (deben
                coincidir para poder fusionarse en ConvFuser).
            grid_size: resolución de la rejilla BEV (grid_size x grid_size celdas), igual
                para ambas ramas.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.camera_encoder = BEVFormerEncoder(embed_dim, bev_channels=bev_channels, grid_size=grid_size)   # rama de cámara
        self.lidar_encoder = LiDARBranch(bev_channels=bev_channels, grid_size=grid_size)   # rama de LiDAR
        self.fuser = ConvFuser(bev_channels)
        self.to_embedding = nn.Linear(bev_channels, embed_dim)

    def forward(self, inputs: dict):
        """Calcula el embedding fusionando las entradas de cámara y LiDAR.

        Args:
            inputs: dict con los tensores de entrada (batch aplanado [K, ...])
                que necesitan tanto la rama de cámara como la de LiDAR.

        Returns:
            [K, embed_dim] — embedding fusionado por muestra.
        """
        camera_bev = self.camera_encoder.bev_features(inputs)   # [K, bev_channels, grid_size, grid_size]
        lidar_bev = self.lidar_encoder(inputs)                  # [K, bev_channels, grid_size, grid_size]
        fused_bev = self.fuser(camera_bev, lidar_bev)           # [K, bev_channels, grid_size, grid_size]
        bev_vector = fused_bev.mean([2, 3])                     # [K, bev_channels] avg sobre grid_size × grid_size (pooling global)

        return self.to_embedding(bev_vector)                    # [K, embed_dim]
