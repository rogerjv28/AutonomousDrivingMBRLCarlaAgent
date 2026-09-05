"""Heads del world model: Decoder (reconstruye la máscara BEV), RewardHead (two-hot), ContinueHead."""
import torch
import torch.nn as nn
from adcarla.utils.norms import RMSNorm

class Decoder(nn.Module):
    """Reconstruye la máscara BEV privilegiada a partir del estado latente (feat -> [B, bev_channels, H, W]).

    Usa una red de ConvTranspose2d para expandir progresivamente el vector latente
    hasta recuperar la resolución espacial del BEV. Si el RSSM puede reconstruir
    el BEV, es porque ha aprendido la geometría real del entorno.
    """

    def __init__(self, feat_dim: int, bev_channels: int, size: int = 128, base_channels: int = 128):
        """Construye la red de decodificación BEV.

        Args:
            feat_dim: dimensión del vector de características latentes de entrada (feat_dim del RSSM).
            bev_channels: canales del mapa BEV de salida (misma semántica que en encoders y world_model).
            size: resolución espacial del BEV de salida (size x size píxeles). Debe ser potencia de 2 >= 16.
            base_channels: canales del tensor espacial inicial antes de las ConvTranspose
                (se divide por 2 en cada capa; última capa → bev_channels).
        """
        super().__init__()
        self.size = size
        self.initial_size = size // 16  # tamaño espacial del tensor antes de las ConvTranspose (ejemplo: 128//16 = 8)
        self.base_channels = base_channels

        # Proyecta el vector latente a un tensor espacial pequeño [B, base_channels, initial_size, initial_size]
        self.to_spatial = nn.Linear(feat_dim, base_channels * self.initial_size * self.initial_size)
        
        # 4 capas ConvTranspose2d que duplican la resolución espacial en cada paso y
        # función de activación SiLU
        self.net = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels // 2, 4, 2, 1), nn.SiLU(),      # x2
            nn.ConvTranspose2d(base_channels // 2, base_channels // 4, 4, 2, 1), nn.SiLU(), # x4
            nn.ConvTranspose2d(base_channels // 4, base_channels // 8, 4, 2, 1), nn.SiLU(), # x8
            nn.ConvTranspose2d(base_channels // 8, bev_channels, 4, 2, 1),                  # x16 -> [B, bev_channels, size, size]
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Reconstruye el mapa BEV desde el estado latente.

        Args:
            feat: vector de características latentes [B, feat_dim].

        Returns:
            Logits del mapa BEV [B, bev_channels, size, size] (sin sigmoid), la pérdida los
            consume con `F.binary_cross_entropy_with_logits`.
        """
        # Proyecta a tensor espacial y reformatea para las ConvTranspose
        x = self.to_spatial(feat).view(-1, self.base_channels, self.initial_size, self.initial_size)
        return self.net(x)   # logits crudos, sigmoid lo aplica la función de pérdida


class RewardHead(nn.Module):
    """Predice la recompensa esperada desde el estado latente usando two-hot encoding.

    Two-hot distribuye la recompensa entre dos bins adyacentes en vez de predecir
    un único float, lo que da una distribución más estable durante el entrenamiento.
    """

    def __init__(self, feat_dim: int, num_bins: int, hidden_dim: int = 256):
        """Construye la cabeza de predicción de recompensa.

        Args:
            feat_dim: dimensión del vector de características latentes de entrada.
            num_bins: número de bins del histograma two-hot (ver configs/base.yaml → world_model.num_bins).
            hidden_dim: dimensión de la capa oculta.
        """
        super().__init__()

        # Una capa linear hacia la hidden layer con activación SiLU y
        # una capa que proyecta hacia el numero de bins marcado.
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), RMSNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, num_bins)
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Devuelve los logits del histograma de recompensa.

        Args:
            feat: estado latente [..., feat_dim].

        Returns:
            Logits two-hot [..., num_bins] (sin softmax, la pérdida los normaliza).
        """
        return self.net(feat)


class ContinueHead(nn.Module):
    """Predice si el episodio continúa (logit de probabilidad de no-done).

    Permite que la imaginación del RSSM tenga en cuenta que el episodio puede terminar
    (por colisión, etc.) y así aprenda a evitarlo durante el entrenamiento en sueño.
    """

    def __init__(self, feat_dim: int, hidden_dim: int = 256):
        """Construye la cabeza de predicción de continuación de episodio.

        Args:
            feat_dim: dimensión del vector de características latentes de entrada.
            hidden_dim: dimensión de la capa oculta.
        """
        super().__init__()

        # Una capa linear hacia la hidden layer con activación SiLU y
        # una capa que proyecta a un número la probabilidad de continuación.
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), RMSNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Devuelve el logit de probabilidad de continuar el episodio.

        Args:
            feat: estado latente [..., feat_dim].

        Returns:
            Logit escalar [...] (sigmoid para obtener probabilidad de continuar).
        """
        # Eliminamos la última dimensión del tensor, que en el caso de tener un número es 1.
        return self.net(feat).squeeze(-1)
