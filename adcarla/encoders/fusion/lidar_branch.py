"""LiDARBranch: features BEV a partir del LiDAR ya rasterizado a BEV [K, Cl, Hb, Wb].
Incluye un rasterizador simple (puntos -> ocupación/altura BEV) para construir la entrada."""
import numpy as np
import torch
import torch.nn as nn


class LiDARBranch(nn.Module):
    """CNN sobre el LiDAR ya rasterizado a BEV -> mapa BEV de características (para fusionar con la cámara)."""

    def __init__(self, in_channels: int = 2, bev_channels: int = 128, grid_size: int = 16):
        """Crea la CNN que procesa el LiDAR rasterizado.

        Args:
            in_channels: canales del BEV de entrada (2 por defecto: ocupación + altura, ver rasterize()).
            bev_channels: canales del mapa BEV de salida.
            grid_size: resolución de salida (grid_size x grid_size) — debe coincidir con la del mapa
                BEV de la cámara, porque luego se fusiona celda a celda en ConvFuser.
        """
        super().__init__()
        self.grid_size = grid_size

        # 2 capas convolucionales con la función de activación Sigmoid Linear Unit y
        # una capa de pooling que hace la media para cada celda de la rejilla BEV.
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(64, bev_channels, 3, 2, 1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((grid_size, grid_size)),
        )

    def forward(self, inputs: dict) -> torch.Tensor:
        """Pasa el LiDAR rasterizado por la CNN.

        Args:
            inputs: dict con la clave "lidar_bev" -> tensor [K, in_channels, size, size].

        Returns:
            [K, bev_channels, grid_size, grid_size] - mapa BEV de características del LiDAR.
        """
        return self.net(inputs["lidar_bev"])    # [K, bev_channels, grid_size, grid_size]

    @staticmethod
    def rasterize(lidar_points: np.ndarray, size: int = 64, range_meters: float = 50.0) -> np.ndarray:
        """Convierte una nube de puntos LiDAR en un histograma BEV de 2 canales.

        Por cada píxel: canal 0 = ocupación (1 si hay algún punto ahí), canal 1 = altura (z)
        máxima de los puntos que caen en esa celda.

        Args:
            lidar_points: (N, 4) puntos (x, y, z, intensity) en metros, relativos al sensor/ego.
            size: resolución de la imagen BEV de salida (size x size).
            range_meters: cobertura, mitad del lado del área cuadrada representada (metros).

        Returns:
            [2, size, size] float32: canal 0 ocupación, canal 1 altura máxima.
        """
        # 2 canales de la representación BEV
        occupancy = np.zeros((size, size), np.float32)
        height_max = np.zeros((size, size), np.float32)

        if lidar_points is None or len(lidar_points) == 0:
            return np.stack([occupancy, height_max], 0)
        
        mpp = (2 * range_meters) / size     # meters per pixel

        # Rasterizamos los puntos reales a la rejilla BEV
        cx = (lidar_points[:, 1] / mpp + size / 2).astype(int)     # y -> col
        cy = (size / 2 - lidar_points[:, 0] / mpp).astype(int)     # x -> row (delante arriba)

        in_bounds = (cx >= 0) & (cx < size) & (cy >= 0) & (cy < size)   # máscara booleana que comprueba que cada punto esté dentro del tamaño de la rejilla BEV
        cx, cy, z = cx[in_bounds], cy[in_bounds], lidar_points[in_bounds, 2]

        occupancy[cy, cx] = 1.0 # escribe el valor 1 en las celdas ocupadas de la rejilla BEV occupancy

        np.maximum.at(height_max, (cy, cx), z)  # escribe la altura máxima correspondiente a las celdas de ocupación en las celdas de la rejilla BEV height_max

        return np.stack([occupancy, height_max], 0) # devuelve los dos canales BEV
