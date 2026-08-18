"""BEVFormerEncoder (reducido): backbone imagen + spatial cross-attention -> BEV -> embedding.
Se usa en la rama visión y como rama de cámara de la fusión (bev_features)."""
import torch
import torch.nn as nn
from ..base import BEVEncoder
from .backbone import ImageBackbone
from .spatial_cross_attn import SpatialCrossAttention

class BEVFormerEncoder(BEVEncoder):
    """Backbone de imagen + cross-attention espacial -> mapa BEV -> embedding (rama de cámara)."""

    def __init__(self, embed_dim: int, bev_channels: int = 128, grid_size: int = 16,
                 num_heads: int = 4, backbone_channels: int = 64):
        """Crea el backbone de imagen y la atención espacial que construyen el mapa BEV.

        Args:
            embed_dim: dimensión final del embedding (la que espera el RSSM).
            bev_channels: dimensión interna del mapa BEV (ver SpatialCrossAttention).
            grid_size: resolución de la rejilla BEV (grid_size x grid_size celdas).
            num_heads: numero de cabezas de atención de SpatialCrossAttention.
            backbone_channels: canales de salida del backbone de imagen (ImageBackbone).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.bev_channels = bev_channels
        self.grid_size = grid_size

        # Módulo CNN Image Backbone
        self.backbone = ImageBackbone(backbone_channels)

        # Módulo Cross Attention
        self.spatial_cross_attention = SpatialCrossAttention(backbone_channels, bev_channels, grid_size, num_heads)

        # Capa linear para transformar la dimensión de la representación bev intermedia al embedding final
        self.to_embedding = nn.Linear(bev_channels, embed_dim)

    def bev_features(self, inputs: dict) -> torch.Tensor:
        """Construye el mapa BEV de características a partir de las imágenes de las cámaras.

        Args:
            inputs: dict con la clave "cameras" -> tensor [K, N, 3, H, W] (K = batch sizze, N = numero de cámaras).

        Returns:
            [K, bev_channels, grid_size, grid_size] — mapa BEV, antes de proyectar a embed_dim.
        """
        cameras = inputs["cameras"] # [K, N, 3, H, W]
        K, num_cameras = cameras.shape[:2]

        # Pasamos al backbone de imagen un batch "plano" de imagenes que contienen las producidas por todas las cámaras
        backbone_features = self.backbone(cameras.reshape(K * num_cameras, *cameras.shape[2:]))   # [K*N, Cf, hf, wf]

        # Reagrupa las features extraídas en una lista "plana" de tokens para pasar al módulo de Spatial Cross Attention
        tokens = backbone_features.flatten(2).transpose(1, 2).reshape(K, -1, backbone_features.shape[1])  # [K, N*hf*wf, Cf]

        return self.spatial_cross_attention(tokens) # [K, bev_channels, grid_size, grid_size]

    def forward(self, inputs: dict) -> torch.Tensor:
        """Calcula el embedding final: mapa BEV -> media espacial -> proyección lineal.

        Args:
            inputs: dict con la clave "cameras".

        Returns:
            [K, embed_dim] — embedding para el RSSM.
        """
        bev = self.bev_features(inputs)

        # Realiza la media de cada canal BEV para obtener el valor que "resume ese canal" y
        # obtener los K embeddings con dimensión correcta para pasar al World Model
        return self.to_embedding(bev.mean([2, 3]))  # [K, embed_dim]
