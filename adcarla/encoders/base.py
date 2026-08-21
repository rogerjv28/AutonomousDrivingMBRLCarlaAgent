"""BEVEncoder: interfaz común de todos los encoders. forward(inputs: dict) -> [K, embed_dim]."""
import torch.nn as nn


class BEVEncoder(nn.Module):
    """Contrato: recibe un dict de tensores (batch aplanado [K, ...]) y devuelve un embedding [K, E]."""
    embed_dim: int

    def forward(self, inputs: dict):
        raise NotImplementedError
