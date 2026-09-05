"""RSSM (DreamerV3): estado latente (h determinista + stoch categórico). observe() / imagine()."""
import torch
import torch.nn as nn
from adcarla.utils.distributions import st_onehot_sample
from adcarla.utils.norms import RMSNorm

class RSSM(nn.Module):
    """Recurrent State Space Model estilo DreamerV3: mantiene el estado latente del mundo en dos
    componentes complementarias, determinista `h` (GRU, acumula el historial) y estocástica
    `stoch` (categórica, captura la incertidumbre del instante actual). Expone `observe()` para 
    entrenar con observaciones reales e `imagine()` para generar trayectorias futuras sin observación.
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

        # Procesa [stoch, action] antes de pasarlo a la GRU (proyección + no linealidad).
        self.pre_action = nn.Sequential(nn.Linear(self.stoch_flat + action_dim, hidden_dim),
                                        RMSNorm(hidden_dim), nn.SiLU())

        # GRU que mantiene el estado determinista h
        self.gru = nn.GRUCell(hidden_dim, deter_dim)

        # Prior: predice stoch solo desde h (sin observación, se usa en imaginación y como referencia KL)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim), RMSNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, self.stoch_flat)
        )

        # Posterior: predice stoch desde h + embedding de los sensores (más informado que el prior)
        self.post_net = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, hidden_dim), RMSNorm(hidden_dim), nn.SiLU(),
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

    def _sample(self, logits_flat: torch.Tensor, stoch_override: torch.Tensor = None):
        """Muestrea el estado estocástico categórico con straight-through.

        Args:
            logits_flat: [B, stoch_flat] logits sin reshape.
            stoch_override: [B, stoch_flat] muestra ya tomada por otro stream. Si se pasa, no se
                muestrea: se devuelve tal cual junto con los logits propios (Raw2Drive Sec. 3.3,
                "eliminación de la aleatoriedad").

        Returns:
            Tupla (stoch_flat [B, stoch_flat], logits [B, stoch_dim, stoch_classes]).
        """
        logits = logits_flat.reshape(-1, self.stoch_dim, self.stoch_classes)
        if stoch_override is not None:
            if stoch_override.shape != logits_flat.shape:
                raise RuntimeError(
                    f"stoch_override deberia tener forma {tuple(logits_flat.shape)} y llegó "
                    f"{tuple(stoch_override.shape)}")
            return stoch_override, logits
        sample = st_onehot_sample(logits)   # [B, stoch_dim, stoch_classes]

        return sample.reshape(logits_flat.shape[0], -1), logits

    def img_step(self, prev: dict, action: torch.Tensor, stoch_override: torch.Tensor = None):
        """Un paso de imaginación: actualiza el estado latente sin observación (solo prior).

        Args:
            prev: estado latente anterior {"h": [B, deter_dim], "stoch": [B, stoch_flat]}.
            action: acción tomada [B, action_dim].
            stoch_override: [B, stoch_flat] muestra estocástica de otro stream; con ella este
                paso no muestrea, adopta la recibida. Es lo que necesita el
                rollout paralelo de los dos world models durante la imaginación (Head Guidance
                por pseudo-deducción, Raw2Drive Sec. 3.3 y Fig. 6c).

        Returns:
            Tupla (estado nuevo, prior_logits [B, stoch_dim, stoch_classes]).
        """
        x = self.pre_action(torch.cat([prev["stoch"], action], dim=-1))  # [B, hidden_dim]
        h = self.gru(x, prev["h"])  # actualiza estado determinista
        stoch, prior_logits = self._sample(self.prior_net(h), stoch_override=stoch_override)

        return {"h": h, "stoch": stoch}, prior_logits

    def obs_step(self, prev: dict, action: torch.Tensor, embed: torch.Tensor,
                 stoch_override: torch.Tensor = None):
        """Un paso de observación: actualiza el estado usando el embedding de los sensores (prior + posterior).

        Args:
            prev: estado latente anterior.
            action: acción tomada [B, action_dim].
            embed: embedding del sensor [B, embed_dim].
            stoch_override: [B, stoch_flat] muestra estocástica de otro stream. Con ella el
                posterior sigue calculando sus logits (los necesita el KL del guidance) pero NO
                muestrea: adopta la muestra recibida. Es como el world model privilegiado consume
                la muestra del alumno en el Rollout Guidance.

        Returns:
            Tupla (estado nuevo, post_logits, prior_logits).
        """
        state, prior_logits = self.img_step(prev, action)
        # El posterior corrige el stoch usando también la observación real
        stoch, post_logits = self._sample(self.post_net(torch.cat([state["h"], embed], dim=-1)),
                                          stoch_override=stoch_override)

        return {"h": state["h"], "stoch": stoch}, post_logits, prior_logits

    def observe(self, embed_seq: torch.Tensor, action_seq: torch.Tensor, state=None,
                stoch_override: torch.Tensor = None):
        """Pasa la secuencia completa por el RSSM con observaciones reales (modo entrenamiento).

        Args:
            embed_seq: embeddings de los sensores [B, T, embed_dim].
            action_seq: acciones [B, T, action_dim].
            state: estado latente inicial.
            stoch_override: [B, T, stoch_flat] muestras estocásticas de otro stream, una por paso
                (ver obs_step). El paso t adopta stoch_override[:, t], que además es el stoch con
                el que se deduce el paso t+1.

        Returns:
            Tupla (states [B,T,...], post_logits [B,T,...], prior_logits [B,T,...], last_state).
        """
        B, T = embed_seq.shape[:2]
        if T == 0:
            raise RuntimeError("observe() requiere al menos un paso de secuencia (T > 0)")
        if state is None:
            state = self.initial_state(B, embed_seq.device)
        if stoch_override is not None and stoch_override.shape[:2] != (B, T):
            raise RuntimeError(
                f"stoch_override deberia tener forma [{B}, {T}, {self.stoch_flat}] y llegó "
                f"{tuple(stoch_override.shape)}")

        h_list, stoch_list, post_logits_list, prior_logits_list = [], [], [], []

        for t in range(T):
            state, post_logits, prior_logits = self.obs_step(
                state, action_seq[:, t], embed_seq[:, t],
                stoch_override=None if stoch_override is None else stoch_override[:, t])
            h_list.append(state["h"])
            stoch_list.append(state["stoch"])
            post_logits_list.append(post_logits)
            prior_logits_list.append(prior_logits)

        stack = lambda l: torch.stack(l, 1)

        return {"h": stack(h_list), "stoch": stack(stoch_list)}, stack(post_logits_list), stack(prior_logits_list), state

    def imagine(self, policy, state: dict, horizon: int, on_step=None):
        """Genera una trayectoria imaginada de longitud horizon sin observaciones reales.

        La política elige acciones a partir del estado latente, y el RSSM avanza
        usando solo el prior (img_step). Se usa para entrenar la política en imaginación.

        La dinámica avanza SIN gradiente: con REINFORCE lo único que sale con
        gradiente son `log_probs` y `entropies`, que es por donde entra el actor.

        Args:
            policy: callable que recibe feat [B, feat_dim] y devuelve (action, log_prob, entropy).
            state: estado latente inicial del que parte la imaginación.
            horizon: número de pasos a imaginar.
            on_step: callable opcional `(action, stoch) -> None` que se invoca tras cada paso.
                Permite avanzar otro world model EN PARALELO con la misma acción y la misma
                muestra estocástica (pseudo-deducción, ver policy/imagination.py).

        Returns:
            Tupla (states, actions, log_probs, entropies) cada uno [B, horizon, ...]; states y
            actions sin gradiente, log_probs y entropies con gradiente hacia la política.
        """
        if horizon == 0:
            raise RuntimeError("imagine() requiere al menos un paso de horizonte (horizon > 0)")
        h_list, stoch_list, action_list, log_prob_list, entropy_list = [], [], [], [], []

        for _ in range(horizon):
            action, log_prob, entropy = policy(self.feat(state).detach())
            with torch.no_grad():
                state, _ = self.img_step(state, action)
            if on_step is not None:
                on_step(action, state["stoch"])
            h_list.append(state["h"])
            stoch_list.append(state["stoch"])
            action_list.append(action)
            log_prob_list.append(log_prob)
            entropy_list.append(entropy)

        stack = lambda l: torch.stack(l, 1)

        return ({"h": stack(h_list), "stoch": stack(stoch_list)}, stack(action_list),
                stack(log_prob_list), stack(entropy_list))
