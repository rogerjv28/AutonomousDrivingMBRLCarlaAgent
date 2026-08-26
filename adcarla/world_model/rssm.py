"""RSSM (DreamerV3): estado latente (h determinista + stoch categórico). observe() / imagine()."""
import torch
import torch.nn as nn
from adcarla.utils.distributions import st_onehot_sample

class RSSM(nn.Module):
    """Recurrent State Space Model estilo DreamerV3: mantiene un estado latente del mundo.

    El estado latente tiene dos componentes complementarias:
    - h (determinista): GRU que acumula el historial.
    - stoch (estocástico): vector categórico muestreado que captura la incertidumbre del momento actual.

    Modos de uso:
    - observe(): entrena el modelo con observaciones reales del sensor (prior vs. posterior).
    - imagine(): genera trayectorias futuras sin observaciones (para entrenar la política con imaginación).
    """

    def __init__(self, action_dim: int, embed_dim: int, deter_dim: int = 512,
                 stoch_dim: int = 32, stoch_classes: int = 32, hidden_dim: int = 512):
        """Construye las redes internas del RSSM.

        Args:
            action_dim: dimensión del vector de acción (entrada junto al estado estocástico).
            embed_dim: dimensión del embedding que produce el encoder de sensores.
            deter_dim: dimensión del estado determinista h (salida de la GRU).
            stoch_dim: número de variables categóricas del estado estocástico.
            stoch_classes: número de clases por variable categórica (stoch_flat = stoch_dim * stoch_classes).
            hidden_dim: dimensión de las capas ocultas de las redes prior, post y pre_action.
        """
        super().__init__()
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.stoch_classes = stoch_classes
        self.stoch_flat = stoch_dim * stoch_classes # dimensión total del estado estocástico aplanado

        # Procesa [stoch, action] antes de pasarlo a la GRU (proyección + no linealidad)
        self.pre_action = nn.Sequential(nn.Linear(self.stoch_flat + action_dim, hidden_dim), nn.SiLU())

        # GRU que mantiene el estado determinista h
        self.gru = nn.GRUCell(hidden_dim, deter_dim)

        # Prior: predice stoch solo desde h (sin observación, se usa en imaginación y como referencia KL)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, self.stoch_flat)
        )

        # Posterior: predice stoch desde h + embedding de los sensores (más informado que el prior)
        self.post_net = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, self.stoch_flat)
        )

    @property
    def feat_dim(self) -> int:
        """Dimensión del vector de características latentes: h concatenado con stoch aplanado."""
        return self.deter_dim + self.stoch_flat

    def initial_state(self, B: int, device) -> dict:
        """Devuelve el estado latente inicial (todo ceros) para un batch de tamaño B."""
        return {"h": torch.zeros(B, self.deter_dim, device=device),
                "stoch": torch.zeros(B, self.stoch_flat, device=device)}

    def feat(self, state: dict) -> torch.Tensor:
        """Concatena h y stoch en un único vector de características [B, feat_dim]."""
        return torch.cat([state["h"], state["stoch"]], dim=-1)

    def _sample(self, logits_flat: torch.Tensor):
        """Muestrea el estado estocástico categórico con straight-through.

        Args:
            logits_flat: [B, stoch_flat] logits sin reshape.

        Returns:
            Tupla (stoch_flat [B, stoch_flat], logits [B, stoch_dim, stoch_classes]).
        """
        logits = logits_flat.reshape(-1, self.stoch_dim, self.stoch_classes)
        sample = st_onehot_sample(logits)   # [B, stoch_dim, stoch_classes]

        return sample.reshape(logits_flat.shape[0], -1), logits

    def img_step(self, prev: dict, action: torch.Tensor):
        """Un paso de imaginación: actualiza el estado latente sin observación (solo prior).

        Args:
            prev: estado latente anterior {"h": [B, deter_dim], "stoch": [B, stoch_flat]}.
            action: acción tomada [B, action_dim].

        Returns:
            Tupla (estado nuevo, prior_logits [B, stoch_dim, stoch_classes]).
        """
        x = self.pre_action(torch.cat([prev["stoch"], action], dim=-1))  # [B, hidden_dim]
        h = self.gru(x, prev["h"])  # actualiza estado determinista
        stoch, prior_logits = self._sample(self.prior_net(h))   # stoch desde prior

        return {"h": h, "stoch": stoch}, prior_logits

    def obs_step(self, prev: dict, action: torch.Tensor, embed: torch.Tensor):
        """Un paso de observación: actualiza el estado usando el embedding de los sensores (prior + posterior).

        Args:
            prev: estado latente anterior.
            action: acción tomada [B, action_dim].
            embed: embedding del sensor [B, embed_dim].

        Returns:
            Tupla (estado nuevo, post_logits, prior_logits).
        """
        state, prior_logits = self.img_step(prev, action)
        # El posterior corrige el stoch usando también la observación real
        stoch, post_logits = self._sample(self.post_net(torch.cat([state["h"], embed], dim=-1)))

        return {"h": state["h"], "stoch": stoch}, post_logits, prior_logits

    def observe(self, embed_seq: torch.Tensor, action_seq: torch.Tensor, state=None):
        """Pasa la secuencia completa por el RSSM con observaciones reales (modo entrenamiento).

        Args:
            embed_seq: embeddings de los sensores [B, T, embed_dim].
            action_seq: acciones [B, T, action_dim].
            state: estado latente inicial.

        Returns:
            Tupla (states [B,T,...], post_logits [B,T,...], prior_logits [B,T,...], last_state).
        """
        B, T = embed_seq.shape[:2]
        if state is None:
            state = self.initial_state(B, embed_seq.device)

        h_list, stoch_list, post_logits_list, prior_logits_list = [], [], [], []

        for t in range(T):
            state, post_logits, prior_logits = self.obs_step(state, action_seq[:, t], embed_seq[:, t])
            h_list.append(state["h"])
            stoch_list.append(state["stoch"])
            post_logits_list.append(post_logits)
            prior_logits_list.append(prior_logits)

        stack = lambda l: torch.stack(l, 1)

        return {"h": stack(h_list), "stoch": stack(stoch_list)}, stack(post_logits_list), stack(prior_logits_list), state

    def imagine(self, policy, state: dict, horizon: int):
        """Genera una trayectoria imaginada de longitud horizon sin observaciones reales.

        La política elige acciones a partir del estado latente, y el RSSM avanza
        usando solo el prior (img_step). Se usa para entrenar la política en imaginación.

        Args:
            policy: callable que recibe feat [B, feat_dim] y devuelve (action, logp, entropy).
            state: estado latente inicial del que parte la imaginación.
            horizon: número de pasos a imaginar.

        Returns:
            Tupla (states, actions, log_probs, entropies) cada uno [B, horizon, ...].
        """
        h_list, stoch_list, action_list, log_prob_list, entropy_list = [], [], [], [], []

        for _ in range(horizon):
            action, log_prob, entropy = policy(self.feat(state))
            state, _ = self.img_step(state, action)
            h_list.append(state["h"])
            stoch_list.append(state["stoch"])
            action_list.append(action)
            log_prob_list.append(log_prob)
            entropy_list.append(entropy)

        stack = lambda l: torch.stack(l, 1)

        return {"h": stack(h_list), "stoch": stack(stoch_list)}, stack(action_list), stack(log_prob_list), stack(entropy_list)
