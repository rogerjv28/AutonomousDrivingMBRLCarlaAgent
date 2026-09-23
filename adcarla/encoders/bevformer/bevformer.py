"""BEVFormerEncoder (reducido): backbone imagen + spatial cross-attention -> BEV -> embedding.
Se usa en la rama visión y como rama de cámara de la fusión (bev_features)."""
import torch
from ..base import BEVEncoder, BEVGridToEmbedding
from .backbone import ImageBackbone
from .spatial_cross_attn import SpatialCrossAttention

class BEVFormerEncoder(BEVEncoder):
    """Backbone de imagen + cross-attention espacial -> mapa BEV -> embedding (rama de cámara)."""

    input_keys = ("cameras", "measurements")

    def __init__(self, embed_dim: int, bev_channels: int = 128, grid_size: int = 16,
                 num_heads: int = 4, backbone_channels: int = 64, standalone: bool = True):
        """Crea el backbone de imagen y la atención espacial que construyen el mapa BEV.

        Args:
            embed_dim: dimensión final del embedding (la que espera el RSSM).
            bev_channels: dimensión interna del mapa BEV (ver SpatialCrossAttention).
            grid_size: resolución de la rejilla BEV (grid_size x grid_size celdas).
            num_heads: numero de cabezas de atención de SpatialCrossAttention.
            backbone_channels: canales de salida del backbone de imagen (ImageBackbone).
            standalone: True si esta instancia se usa como encoder completo (rama visión). False en
                la rama de cámara anidada en FusionEncoder, que solo llama a bev_features().
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.bev_channels = bev_channels
        self.grid_size = grid_size
        self.grid_channels = bev_channels   # canales de la rejilla que expone bev_features()
        self.standalone = standalone

        # Módulo CNN Image Backbone
        self.backbone = ImageBackbone(backbone_channels)

        # Módulo Cross Attention
        self.spatial_cross_attention = SpatialCrossAttention(backbone_channels, bev_channels, grid_size, num_heads)

        # Cola "suelta" del encoder: compresión rejilla BEV -> embedding (sin promediar el
        # espacio) + medidas del ego. Anidado en FusionEncoder no se crea ninguna de las dos:
        # la fusión comprime su propia rejilla (la ya fusionada) y combina las medidas una sola vez.
        if standalone:
            self.to_embedding = BEVGridToEmbedding(bev_channels, grid_size, embed_dim)
            self._init_measurement_head(embed_dim)  # velocidad + target point + comando

    def bev_features(self, inputs: dict) -> torch.Tensor:
        """Construye el mapa BEV de características a partir de las imágenes de las cámaras.

        Args:
            inputs: dict con la clave "cameras" -> tensor [K, N, 3, H, W] (K = batch size, N = numero de cámaras).

        Returns:
            [K, bev_channels, grid_size, grid_size] — mapa BEV, antes de proyectar a embed_dim.
        """
        if "cameras" not in inputs:
            raise RuntimeError("BEVFormerEncoder.bev_features() requiere la clave 'cameras' en inputs")
        cameras = inputs["cameras"]  # [K, N, 3, H, W]
        K, num_cameras = cameras.shape[:2]

        # Pasamos al backbone de imagen un batch "plano" de imagenes que contienen las producidas por todas las cámaras
        backbone_features = self.backbone(cameras.reshape(K * num_cameras, *cameras.shape[2:]))   # [K*N, Cf, hf, wf]

        # Reagrupa las features extraídas en una lista "plana" de tokens para pasar al módulo de Spatial Cross Attention
        tokens = backbone_features.flatten(2).transpose(1, 2).reshape(K, -1, backbone_features.shape[1])  # [K, N*hf*wf, Cf]

        return self.spatial_cross_attention(tokens) # [K, bev_channels, grid_size, grid_size]

    def forward_with_grid(self, inputs: dict):
        """Embedding + rejilla BEV, con el guard de `standalone`.

        Args:
            inputs: dict con la clave "cameras".

        Returns:
            Tupla (embedding [K, embed_dim], rejilla [K, bev_channels, grid_size, grid_size]).

        Raises:
            RuntimeError: si la instancia no es standalone (rama de cámara de FusionEncoder).
        """
        if not self.standalone:
            raise RuntimeError(
                "BEVFormerEncoder.forward() no existe en una instancia con standalone=False "
                "(rama de cámara de FusionEncoder): no tiene ni compresión a embedding ni head "
                "de medidas. Usa bev_features() y comprime la rejilla ya fusionada.")
        return super().forward_with_grid(inputs)
