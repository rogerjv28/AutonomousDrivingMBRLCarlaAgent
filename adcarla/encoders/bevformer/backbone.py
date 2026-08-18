"""ImageBackbone: extractor ligero de features de imagen (CNN). Devuelve [K, Cf, hf, wf]."""
import torch.nn as nn

class ImageBackbone(nn.Module):
    def __init__(self, out_channels: int = 64):
        super().__init__()
        # 3 capas convolucionales que reducen el tamaño a la mitad y duplican las dimensiones
        # con la función de activación Sigmoid Linear Unit
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(64, out_channels, 3, 2, 1), nn.SiLU(),
        )
        self.out_channels = out_channels

    def forward(self, x):
        return self.net(x)
