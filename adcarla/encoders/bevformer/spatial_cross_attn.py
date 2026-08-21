"""SpatialCrossAttention: BEV queries que atienden a los tokens de las cámaras (versión reducida,
sin geometría deformable). Devuelve un mapa BEV [K, d, G, G]."""
import torch
import torch.nn as nn


class SpatialCrossAttention(nn.Module):
    """Rejilla de queries BEV que atienden (cross-attention) a los tokens de imagen de las cámaras."""

    def __init__(self, token_dim: int, bev_channels: int = 128, grid_size: int = 16, num_heads: int = 4):
        """Crea las queries BEV aprendibles y las capas de atención.

        Args:
            token_dim: tamaño de cada token de imagen que entra en forward().
            bev_channels: dimensión interna usada por la atención (y del mapa BEV de salida).
            grid_size: resolución de la rejilla BEV de salida (grid_size x grid_size celdas).
            num_heads: numero de cabezas de nn.MultiheadAttention.
        """
        super().__init__()
        self.grid_size = grid_size
        self.bev_channels = bev_channels

        # Crea los parámetros de las queries aprendibles y las inicializa con valores pequeños,
        # uno por cada celda de salida y cada canal BEV
        self.bev_queries = nn.Parameter(torch.randn(grid_size * grid_size, bev_channels) * 0.02)

        # Capa linear para transformar la dimensión de los tokens de entrada a la representación bev intermedia
        self.token_projection = nn.Linear(token_dim, bev_channels)

        # 
        self.cross_attention = nn.MultiheadAttention(bev_channels, num_heads, batch_first=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Hace que cada celda BEV atienda a todos los tokens de imagen y construye el mapa BEV.

        Args:
            tokens: [K, T, token_dim] - tokens de imagen (T = tokens totales de todas las cámaras).

        Returns:
            [K, bev_channels, grid_size, grid_size] - mapa BEV de características.
        """
        K = tokens.shape[0]
        projected_tokens = self.token_projection(tokens) # convierte tokens a dimensión adecuada
        queries = self.bev_queries.unsqueeze(0).expand(K, -1, -1) # añadimos la dimensión del batch a las queries para poder operar

        # Ejecutamos el mecanismo de atención lanzando las queries
        bev, _ = self.cross_attention(queries, projected_tokens, projected_tokens)   # [K, grid_size*grid_size, bev_channels]

        # Transformamos a la forma [K, bev_channels, grid_size, grid_size] para que este mapa BEV pueda ser tratado
        return bev.transpose(1, 2).reshape(K, self.bev_channels, self.grid_size, self.grid_size)
    