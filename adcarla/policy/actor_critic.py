"""ActorCritic: actor discreto (one-hot straight-through) + critic categórico (two-hot).

Ambos operan sobre el feature latente del RSSM (feat = [h, stoch]), nunca sobre observaciones
crudas: se entrenan enteramente "imaginando" rollouts dentro del world model (ver imagination.py).
"""
import torch
import torch.nn as nn
from adcarla.utils.distributions import make_bins, from_probs, symexp


class Actor(nn.Module):
    """Política discreta: MLP feat -> logits de acción, muestreo one-hot categórico
    con straight-through estimator (permite retropropagar a través del muestreo discreto)."""

    def __init__(self, feat_dim: int, num_actions: int, hidden_dim: int = 256):
        """Crea el MLP del actor.

        Args:
            feat_dim: dimensión del feature latente del RSSM (entrada).
            num_actions: número de acciones discretas (salida).
            hidden_dim: anchura de las capas ocultas.
        """
        super().__init__()
        # 3 capas lineales completamente conectadas con función de activación SiLU entre capas
        self.net = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, num_actions))

    def forward(self, feat: torch.Tensor):
        """Muestrea una acción para entrenar (rollout de imaginación).

        Args:
            feat: [..., feat_dim] feature latente del RSSM.

        Returns:
            action: [..., num_actions] one-hot (con gradiente vía straight-through).
            entropy: [...] entropía de la distribución (para el bonus de exploración).
        """
        logits = self.net(feat)
        dist = torch.distributions.OneHotCategorical(logits=logits)
        sample = dist.sample()

        # Straight-through: en el forward pasamos el one-hot muestreado (sample),
        # pero el gradiente fluye como si fuera dist.probs (sample - sample.detach() = 0 en valor,
        # pero deja pasar el gradiente de probs). Así se puede derivar a través de una acción discreta.
        action = sample + dist.probs - dist.probs.detach()

        return action, dist.entropy()

    def act(self, feat: torch.Tensor) -> torch.Tensor:
        """Acción para inferencia/evaluación (sin muestreo, sin gradiente).

        Returns:
            Tensor de índices enteros [...], el caller hace .item() si necesita un int escalar.
        """
        with torch.no_grad():
            return self.net(feat).argmax(-1)


class Critic(nn.Module):
    """Valor de estado categórico (two-hot, estilo DreamerV3): en vez de regresar el valor
    escalar directamente (MSE, inestable con escalas de recompensa muy dispares), predice una
    distribución sobre bins fijos y reconstruye el valor esperado a partir de ella."""

    def __init__(self, feat_dim: int, num_bins: int, hidden_dim: int = 256):
        """Crea el MLP del critic.

        Args:
            feat_dim: dimensión del feature latente del RSSM (entrada).
            num_bins: número de bins de la distribución two-hot de valor (salida).
            hidden_dim: anchura de las capas ocultas.
        """
        super().__init__()

        # 2 capas lineales con función de activación SiLU en medio
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, num_bins)
        )

        self.register_buffer("bins", make_bins(num_bins))   # valores fijos (en espacio symlog) que representa cada bin

    def forward(self, feat):
        """Logits de la distribución two-hot de valor (sin normalizar). [..., num_bins]. Se usa
        directamente en la pérdida (two_hot_loss espera logits, no probabilidades)."""
        return self.net(feat)

    def value(self, feat) -> torch.Tensor:
        """Valor esperado en escala real (para bootstrapping de λ-returns, no para la pérdida).

        Convierte los logits en probabilidades, calcula la esperanza sobre los bins (en espacio
        symlog) y deshace la compresión symlog con symexp para recuperar la escala original.
        """
        probs = torch.softmax(self.net(feat), dim=-1)
        return symexp(from_probs(probs, self.bins))
