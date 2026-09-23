"""ActorCritic: actor discreto entrenado con REINFORCE + critic categórico (two-hot).

Ambos operan sobre el feature latente del RSSM (feat = [h, stoch]), nunca sobre observaciones
crudas: se entrenan enteramente "imaginando" rollouts dentro del world model (ver imagination.py).
"""
import copy

import torch
import torch.nn as nn
from adcarla.utils.distributions import make_bins, from_probs, symexp, unimix_logits
from adcarla.utils.norms import RMSNorm


class ReturnNormalizer(nn.Module):
    """Escala de los retornos por percentiles (DreamerV3 Ec. 7): `S = EMA(perc95 - perc5, 0.99)`.

    La ventaja del actor se divide por `max(1, S)`.
    """

    PERCENTILES = (0.05, 0.95)
    LIMIT = 1.0

    def __init__(self, decay: float):
        """
        Args:
            decay: factor de la media móvil exponencial (`policy.return_norm_decay`, 0.99).
        """
        super().__init__()
        self.decay = float(decay)
        # Buffers (no parámetros): viajan en el state_dict, así que --resume no reinicia la escala.
        self.register_buffer("scale", torch.zeros(()))
        self.register_buffer("started", torch.zeros((), dtype=torch.bool))
        self.register_buffer("percentiles", torch.tensor(self.PERCENTILES))

    @torch.no_grad()
    def update(self, returns: torch.Tensor) -> torch.Tensor:
        """Actualiza la EMA con el batch de retornos y devuelve el denominador `max(1, S)`.

        Args:
            returns: λ-returns del rollout imaginado [...]; se aplanan para los percentiles.

        Returns:
            Escalar `max(1, S)` por el que dividir la ventaja.
        """
        percentiles = torch.quantile(returns.detach().flatten().float(), self.percentiles)
        span = (percentiles[1] - percentiles[0]).to(self.scale.dtype)

        # La primera actualización fija S al rango observado: arrancar la EMA en 0 dejaría ~100
        # updates con el denominador pegado al límite y ventajas sin normalizar.
        self.scale.copy_(span if not bool(self.started) else
                         self.decay * self.scale + (1.0 - self.decay) * span)
        self.started.fill_(True)

        return self.scale.clamp(min=self.LIMIT)


class Actor(nn.Module):
    """Política discreta: MLP feat -> logits de acción, muestreo one-hot categórico.

    El gradiente entra por `log π(a|s)` (REINFORCE) y por la entropía, no por la acción
    muestreada: `forward` devuelve la acción ya desconectada del grafo.
    """

    def __init__(self, feat_dim: int, num_actions: int, return_norm_decay: float,
                 hidden_dim: int = 256):
        """Crea el MLP del actor y su normalizador de retornos.

        Args:
            feat_dim: dimensión del feature latente del RSSM (entrada).
            num_actions: número de acciones discretas (salida).
            return_norm_decay: decay de la EMA de `ReturnNormalizer` (`policy.return_norm_decay`).
            hidden_dim: anchura de las capas ocultas.
        """
        super().__init__()
        # 3 capas lineales completamente conectadas con función de activación SiLU entre capas
        self.net = nn.Sequential(nn.Linear(feat_dim, hidden_dim), RMSNorm(hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, hidden_dim), RMSNorm(hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, num_actions))
        self.return_norm = ReturnNormalizer(return_norm_decay)

    def logits(self, feat: torch.Tensor) -> torch.Tensor:
        """Logits sin normalizar de la distribución de acciones [..., num_actions].

        Los necesita la destilación (CE contra la distribución del profesor), que no muestrea.
        """
        return self.net(feat)

    def forward(self, feat: torch.Tensor):
        """Muestrea una acción de la política (rollout de imaginación y recogida real).

        Args:
            feat: [..., feat_dim] feature latente del RSSM.

        Returns:
            action: [..., num_actions] one-hot muestreado, sin gradiente.
            log_prob: [...] log π(a|s) de la acción muestreada (con gradiente: es el estimador
                REINFORCE).
            entropy: [...] entropía de la distribución (regularizador de exploración).
        """
        # Unimix del 1 % (DreamerV3): la política nunca asigna probabilidad exactamente 0, así
        # que log π no se va a -inf y queda un suelo de exploración.
        dist = torch.distributions.OneHotCategorical(logits=unimix_logits(self.logits(feat)))
        action = dist.sample()   # sample() ya sale del grafo: REINFORCE no deriva por la acción

        return action, dist.log_prob(action), dist.entropy()

    def act(self, feat: torch.Tensor) -> torch.Tensor:
        """Acción para inferencia/evaluación (sin muestreo, sin gradiente).

        Args:
            feat: [..., feat_dim] feature latente del RSSM.

        Returns:
            Tensor de índices enteros [...], el caller hace .item() si necesita un int escalar.
        """
        with torch.no_grad():
            return self.net(feat).argmax(-1)


class Critic(nn.Module):
    """Valor de estado categórico (two-hot, estilo DreamerV3): en vez de regresar el valor
    escalar directamente (MSE es inestable con escalas de recompensa muy dispares), predice una
    distribución sobre bins fijos y reconstruye el valor esperado a partir de ella.

    Lleva además un target lento (`slow`): una copia de la red actualizada por media móvil
    exponencial hacia la que se regulariza el critic (DreamerV3, "Critic learning"). Sin él, el
    critic regresa targets que dependen de sus propias predicciones y el bootstrap se persigue
    a sí mismo.
    """

    def __init__(self, feat_dim: int, num_bins: int, bin_min: float, bin_max: float,
                 ema_decay: float, hidden_dim: int = 256):
        """Crea el MLP del critic y su target lento.

        Args:
            feat_dim: dimensión del feature latente del RSSM (entrada).
            num_bins: número de bins de la distribución two-hot de valor (salida).
            bin_min: extremo inferior de los bins, en espacio symlog (`policy.critic_bin_min`).
            bin_max: extremo superior (`policy.critic_bin_max`). El rango NO coincide con el de la
                cabeza de reward: un λ-return acumula cientos de ticks descontados.
            ema_decay: decay del target lento (`policy.critic_ema_decay`, 0.98).
            hidden_dim: anchura de las capas ocultas.
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), RMSNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, num_bins)
        )

        self.ema_decay = float(ema_decay)
        self.slow = copy.deepcopy(self.net)     # arranca como copia exacta del critic
        self.slow.requires_grad_(False)         # se mueve solo por EMA, nunca por gradiente

        self.register_buffer("bins", make_bins(num_bins, bin_min, bin_max))   # valores (symlog) de cada bin

    def forward(self, feat):
        """Logits de la distribución two-hot de valor (sin normalizar). [..., num_bins]. Se usa
        directamente en la pérdida (two_hot_loss espera logits, no probabilidades)."""
        return self.net(feat)

    def _expected_value(self, logits: torch.Tensor) -> torch.Tensor:
        """Esperanza sobre los bins (espacio symlog) deshecha con symexp: valor en escala real."""
        return symexp(from_probs(torch.softmax(logits, dim=-1), self.bins))

    def value(self, feat) -> torch.Tensor:
        """Valor esperado en escala real (para bootstrapping de λ-returns, no para la pérdida)."""
        return self._expected_value(self.net(feat))

    @torch.no_grad()
    def slow_value(self, feat) -> torch.Tensor:
        """Valor esperado según el target lento: el objetivo del regularizador del critic."""
        return self._expected_value(self.slow(feat))

    @torch.no_grad()
    def update_slow(self):
        """Un paso de EMA del target lento hacia los pesos actuales. Lo llama el trainer después
        de cada `opt_critic.step()`."""
        for fast_param, slow_param in zip(self.net.parameters(), self.slow.parameters()):
            slow_param.mul_(self.ema_decay).add_(fast_param, alpha=1.0 - self.ema_decay)
