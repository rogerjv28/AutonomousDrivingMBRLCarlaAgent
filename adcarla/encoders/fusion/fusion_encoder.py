"""FusionEncoder: BEVFormer (cámara, misma arquitectura que la rama visión) + LiDARBranch + ConvFuser
-> embedding. NO comparte pesos con la rama visión (cada rama entrena su propio encoder)."""
import torch
from ..base import BEVEncoder, BEVGridToEmbedding
from ..bevformer.bevformer import BEVFormerEncoder
from .lidar_branch import LiDARBranch
from .fuser import ConvFuser

class FusionEncoder(BEVEncoder):
    """Encoder de la rama de fusión (cámara + LiDAR): dos ramas BEV independientes -> ConvFuser -> embedding.

    Combina un encoder BEVFormer (cámara) y la LiDARBranch (LiDAR rasterizado),
    ambos con la misma resolución de rejilla BEV, fusiona sus mapas BEV con
    ConvFuser y comprime la rejilla fusionada a un embedding. No comparte
    pesos con la rama vision-only.
    """

    input_keys = ("cameras", "lidar_bev", "measurements")

    def __init__(self, embed_dim: int, bev_channels: int = 128, grid_size: int = 16,
                 num_heads: int = 4, backbone_channels: int = 64, lidar_in_channels: int = 2):
        """Crea las dos ramas (cámara, LiDAR), el fusor y la proyección final.

        Args:
            embed_dim: dimensión del embedding de salida que consume el resto
                del modelo (world model).
            bev_channels: canales internos de los mapas BEV de cámara y LiDAR (deben
                coincidir para poder fusionarse en ConvFuser).
            grid_size: resolución de la rejilla BEV (grid_size x grid_size celdas), igual
                para ambas ramas.
            num_heads: cabezas de la spatial cross-attention de la rama de cámara. Debe
                coincidir con el de la rama visión.
            backbone_channels: canales de salida del backbone de imagen de la rama de cámara.
                Debe coincidir con el de la rama visión.
            lidar_in_channels: canales del BEV rasterizado del LiDAR (ocupación + altura).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.grid_channels = bev_channels   # canales de la rejilla que expone bev_features()
        # IMPORTANTE: se propagan TODOS los parametros de camara (incluidos num_heads y
        # backbone_channels); standalone=False.
        self.camera_encoder = BEVFormerEncoder(embed_dim, bev_channels=bev_channels, grid_size=grid_size,
                                               num_heads=num_heads, backbone_channels=backbone_channels,
                                               standalone=False)   # rama de cámara
        self.lidar_encoder = LiDARBranch(in_channels=lidar_in_channels, bev_channels=bev_channels,
                                          grid_size=grid_size)   # rama de LiDAR
        self.fuser = ConvFuser(bev_channels)
        self.to_embedding = BEVGridToEmbedding(bev_channels, grid_size, embed_dim)
        self._init_measurement_head(embed_dim)  # velocidad + target point + comando

    def bev_features(self, inputs: dict) -> torch.Tensor:
        """Rejilla BEV fusionada (cámara + LiDAR), antes de comprimirla a embedding.

        Args:
            inputs: dict con los tensores de entrada (batch aplanado [K, ...])
                que necesitan tanto la rama de cámara como la de LiDAR.

        Returns:
            [K, bev_channels, grid_size, grid_size].
        """
        camera_bev = self.camera_encoder.bev_features(inputs)   # [K, bev_channels, grid_size, grid_size]
        lidar_bev = self.lidar_encoder(inputs)                  # [K, bev_channels, grid_size, grid_size]
        return self.fuser(camera_bev, lidar_bev)                # [K, bev_channels, grid_size, grid_size]
