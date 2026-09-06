"""BEVEncoder: interfaz común de todos los encoders. forward(inputs: dict) -> [K, embed_dim].

También vive aquí `BEVGridToEmbedding`, la compresión rejilla → embedding que comparten los tres
encoders.
"""
import torch
import torch.nn as nn

from .measurements import MeasurementEncoder


def _grid_side_after_two_convs(grid_size: int) -> int:
    """Lado de la rejilla tras las dos Conv2d(3, stride=2, padding=1) de BEVGridToEmbedding (16 → 4)."""
    side = grid_size
    for _ in range(2):
        side = (side - 1) // 2 + 1
    return side


class BEVGridToEmbedding(nn.Module):
    """Comprime la rejilla BEV [K, C, G, G] a un embedding [K, embed_dim] (D3).

    Dos convoluciones stride 2 (16 → 8 → 4) + flatten + Linear.
    """

    def __init__(self, in_channels: int, grid_size: int, embed_dim: int):
        """
        Args:
            in_channels: canales de la rejilla BEV de entrada.
            grid_size: lado de la rejilla BEV de entrada (G x G).
            embed_dim: dimensión del embedding de salida.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, 3, 2, 1), nn.SiLU(),
        )
        side = _grid_side_after_two_convs(grid_size)
        self.fc = nn.Linear(in_channels * side * side, embed_dim)

    def forward(self, bev_grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bev_grid: [K, in_channels, G, G].

        Returns:
            [K, embed_dim].
        """
        return self.fc(self.net(bev_grid).flatten(1))


class BEVEncoder(nn.Module):
    """Contrato: recibe un dict de tensores (batch aplanado [K, ...]) y devuelve un embedding [K, E]."""
    embed_dim: int      # obligatorio: cada subclase debe asignar self.embed_dim en __init__
    grid_channels: int  # canales de la rejilla que devuelve bev_features() (los consume el guidance)
    # Claves de `inputs` que este encoder lee de verdad (incluida "measurements"). WorldModel._encode_seq
    # filtra por esta lista antes de trocear la secuencia, para no arrastrar tensores que el encoder
    # ignora (p.ej. "bev", que solo lee el profesor) a través del troceado por memoria.
    input_keys: tuple = ("measurements",)

    def bev_features(self, inputs: dict) -> torch.Tensor:
        """Rejilla BEV de características, antes de comprimirla a embedding.

        Es lo que alinea el guidance por celda entre profesor y alumno (Ec. 1 de Raw2Drive):
        `forward()` la reutiliza, así que las dos rutas no pueden divergir.

        Args:
            inputs: dict de tensores con batch aplanado [K, ...]; las claves dependen del encoder.

        Returns:
            [K, grid_channels, G, G] — rejilla BEV, con G = `bev.grid_size` del config.
        """
        raise NotImplementedError

    def forward_with_grid(self, inputs: dict):
        """Embedding y rejilla BEV en una sola pasada.

        El rollout guidance necesita las dos cosas del mismo forward: recalcular la rejilla aparte
        costaría otro pase completo del encoder (la parte cara del alumno).

        Args:
            inputs: dict de tensores con batch ya aplanado [K, ...].

        Returns:
            Tupla (embedding [K, embed_dim], rejilla [K, grid_channels, G, G]).
        """
        grid = self.bev_features(inputs)
        return self._combine_with_measurements(self.to_embedding(grid), inputs), grid

    def forward(self, inputs: dict) -> torch.Tensor:
        """Procesa el dict de observaciones y devuelve el embedding [K, embed_dim].

        Args:
            inputs: dict de tensores con batch ya aplanado [K, ...]; las claves dependen del encoder.

        Returns:
            [K, embed_dim] — embedding del encoder listo para el RSSM.
        """
        return self.forward_with_grid(inputs)[0]

    # ---- MEDIDAS DEL EGO ----
    # Helper común a los tres encoders
    def _init_measurement_head(self, embed_dim: int, hidden_dim: int = 64):
        """Crea el `MeasurementEncoder` y la capa que combina su salida con el embedding BEV.

        Llamar desde `__init__`, después de fijar `self.embed_dim`.
        """
        self.measurement_encoder = MeasurementEncoder(hidden_dim)
        self.embed_combiner = nn.Linear(embed_dim + hidden_dim, embed_dim)

    def _combine_with_measurements(self, bev_embed: torch.Tensor, inputs: dict) -> torch.Tensor:
        """Concatena el embedding BEV con el de las medidas del ego y proyecta a `embed_dim`.

        Args:
            bev_embed: [K, embed_dim], embedding ya calculado por la parte BEV de la subclase.
            inputs: dict de entrada del encoder, debe traer la clave "measurements" ([K, 9]).

        Returns:
            [K, embed_dim].
        """
        if "measurements" not in inputs:
            raise RuntimeError(f"{type(self).__name__}.forward() requiere la clave 'measurements' en inputs")
        measurements_embed = self.measurement_encoder(inputs["measurements"])
        return self.embed_combiner(torch.cat([bev_embed, measurements_embed], dim=-1))
