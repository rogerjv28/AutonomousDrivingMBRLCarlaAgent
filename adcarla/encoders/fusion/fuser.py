"""ConvFuser: fusiona en BEV las features de cámara (BEVFormer) y de LiDAR."""
import torch
import torch.nn as nn


class ConvFuser(nn.Module):
    """Fusiona dos mapas BEV (obtenidos de cámara y LiDAR) en uno solo mediante concatenación + convoluciones.

    Recibe dos tensores BEV con la misma forma espacial [K, bev_channels, G, G] (uno por
    cámara, otro por LiDAR), los concatena por canales y aplica dos capas
    convolucionales 3x3 con activación SiLU para aprender a combinarlos en
    un único mapa BEV fusionado de vuelta al número de canales "bev_channels".
    K = tamaño del batch, G = resolución de la rejilla BEV.
    """

    def __init__(self, bev_channels: int = 128):
        """Crea las capas del fusor.

        Args:
            bev_channels: número de canales de cada mapa BEV de entrada (cámara y
                LiDAR). La entrada concatenada tiene "2*bev_channels" canales, la salida
                fusionada vuelve a tener "bev_channels" canales.
        """
        super().__init__()
        # 2 capas convolucionales con la función de activación Sigmoid Linear Unit
        # que devuelven una salida con los canales esperados
        self.net = nn.Sequential(nn.Conv2d(2 * bev_channels, bev_channels, 3, 1, 1), nn.SiLU(),
                                 nn.Conv2d(bev_channels, bev_channels, 3, 1, 1), nn.SiLU())

    def forward(self, camera_bev, lidar_bev):
        """Fusiona los dos mapas BEV en uno solo.

        Args:
            camera_bev: mapa BEV de cámara, forma [K, bev_channels, G, G].
            lidar_bev: mapa BEV de LiDAR, misma forma [K, bev_channels, G, G].

        Returns:
            Mapa BEV fusionado, forma [K, bev_channels, G, G].
        """
        # Concatena por canales -> [K, 2*bev_channels, H, W] y pasa por las convoluciones.
        return self.net(torch.cat([camera_bev, lidar_bev], dim=1))
