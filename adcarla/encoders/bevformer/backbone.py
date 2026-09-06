"""ImageBackbone: extractor ligero de features de imagen (CNN). Devuelve [K, Cf, hf, wf]."""
import torch.nn as nn

#: Factor de reducción espacial total del backbone (4 bloques stride 2).
REDUCTION = 16


class ImageBackbone(nn.Module):
    """CNN ligera de cuatro capas Conv2d+SiLU; reduce H y W en factor 16 -> [K, out_channels, Hf, Wf]."""

    def __init__(self, out_channels: int = 64):
        super().__init__()
        # 4 capas convolucionales que reducen el tamaño a la mitad con la función de activación
        # Sigmoid Linear Unit.
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 64, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(64, out_channels, 3, 2, 1), nn.SiLU(),
        )
        self.out_channels = out_channels

    def forward(self, x):
        return self.net(x)
