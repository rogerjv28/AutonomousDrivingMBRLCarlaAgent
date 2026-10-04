"""Utilidades de rollout: conversión de observaciones de CARLA y agente de conducción en lazo cerrado."""
import random

import numpy as np
import torch
import torch.nn.functional as F
from adcarla.encoders.fusion.lidar_branch import LiDARBranch
from adcarla.training.replay_buffer import normalize_sensor_value


def obs_to_step(obs: dict, config: dict) -> dict:
    """Convierte una observación raw de CarlaEnv en un dict numpy listo para el encoder y el decoder.

    Args:
        obs: dict de CarlaEnv (con claves "bev_privileged", "cameras", "lidar", "measurements").
        config: dict de configuración completa.

    Returns:
        Dict de arrays numpy con las claves disponibles según la rama y los sensores activos.
    """
    step = {}

    # Máscara BEV privilegiada: solo disponible cuando privileged_bev=True en la configuración.
    # Se guarda en uint8 porque la máscara es binaria y en float32 ocuparía cuatro veces más
    # memoria en el replay. `sample()` la convierte a float.
    if obs.get("bev_privileged") is not None:
        step["bev"] = np.asarray(obs["bev_privileged"], dtype=np.uint8)    # [Cb, Hb, Wb]

    # Medidas del ego (velocidad, punto objetivo y comando). Las calcula CarlaEnv.
    if obs.get("measurements") is not None:
        step["measurements"] = np.asarray(obs["measurements"], dtype=np.float32)    # [9]

    # Cámaras: se transpone de HWC a CHW (formato PyTorch) y se mantienen en uint8 [0, 255].
    # La normalización a [0, 1] la hace `sample()`.
    cams = obs.get("cameras")
    if cams:
        cam_images = [np.transpose(c, (2, 0, 1)) for c in cams.values() if c is not None]
        if cam_images:
            step["cameras"] = np.stack(cam_images, 0).astype(np.uint8)    # [N, 3, H, W] (N cámaras)

    # LiDAR: solo rama fusión, rasteriza la nube de puntos a una proyección BEV en 2 canales
    if obs.get("lidar") is not None:
        lidar_range_m = float(config.get("bev", {}).get("range_meters", 50.0))
        step["lidar_bev"] = LiDARBranch.rasterize(np.asarray(obs["lidar"]), range_meters=lidar_range_m)  # [2, Hb, Wb]

    return step


def action_to_onehot(action_idx, num_actions: int) -> np.ndarray:
    """Codifica la acción previa como one-hot float32 para `step_data`, igual que `RolloutAgent.prev`.

    Con `action_idx=None` devuelve el vector nulo, porque antes del primer paso del episodio no hay
    acción previa. Cada observación o_t se guarda junto a a_{t-1}, no junto a a_t.
    """
    onehot = np.zeros(num_actions, dtype=np.float32)
    if action_idx is not None:
        onehot[action_idx] = 1.0
    return onehot


def _batchify(step: dict, device) -> dict:
    """Añade dimensión de batch (B=1) a cada array del step, lo convierte a tensor float32 y lo
    normaliza con `normalize_sensor_value`, la misma función que usa `sample()`. Así el encoder
    recibe las cámaras en [0, 1] tanto al actuar como al entrenar."""
    return {k: normalize_sensor_value(k, torch.as_tensor(v, dtype=torch.float32, device=device).unsqueeze(0))
            for k, v in step.items()}


class RolloutAgent:
    """Agente de conducción en lazo cerrado que mantiene el estado latente del RSSM entre steps.

    Gestiona el ciclo observación → embed → obs_step → actor durante la recogida de observaciones reales.
    No interviene en el entrenamiento por imaginación (ver imagination.py).
    """

    def __init__(self, wm, actor, config: dict, device="cuda"):
        """
        Args:
            wm: WorldModel que provee el encoder, el RSSM y num_actions.
            actor: Actor cuya política se usa para elegir acciones.
            config: configuración completa del experimento.
            device: dispositivo PyTorch donde residen los tensores del estado.
        """
        self.wm = wm
        self.actor = actor
        self.config = config
        self.device = device
        self.reset()

    def reset(self):
        """Reinicia el estado latente y la acción previa al inicio de un episodio."""
        self.state = self.wm.rssm.initial_state(1, self.device)     # {"h": [1, deter], "stoch": [1, stoch_flat]}
        self.prev = torch.zeros(1, self.wm.num_actions, device=self.device)  # acción previa como one-hot

    @torch.no_grad()
    def act(self, obs: dict, greedy: bool = True) -> int:
        """Codifica la observación, actualiza el estado latente y devuelve la acción elegida.

        Args:
            obs: observación de CarlaEnv.
            greedy: True → argmax (evaluación), False → muestreo categórico (exploración).

        Returns:
            Índice entero de la acción elegida (compatible con env.step()).
        """
        obs_batch = _batchify(obs_to_step(obs, self.config), self.device)           # dict de arrays numpy
        obs_embed = self.wm.encoder(obs_batch)  # [1, embed_dim]                    # embedding generado por el encoder para el RSSM
        self.state, _, _ = self.wm.rssm.obs_step(self.state, self.prev, obs_embed)  # estado latente del RSSM actualizado
        feat = self.wm.rssm.feat(self.state)    # [1, feat_dim]                     # estado latente concatenado

        if greedy:
            action_idx = int(self.actor.act(feat).item())   # argmax sin gradiente
        else:
            # La acción se muestrea con el mismo método que usa la imaginación (devuelve un one-hot).
            action, _, _ = self.actor(feat)
            action_idx = int(action.argmax(-1).item())

        # Guarda la acción como one-hot para condicionar el siguiente obs_step del RSSM
        self.prev = F.one_hot(torch.tensor([action_idx], device=self.device), self.wm.num_actions).float()
        return action_idx


class RandomAgent:
    """Agente de exploración pura: elige acciones uniformes y no usa ninguna red. Se usa en el
    prefill del replay buffer. Tiene la misma interfaz reset()/act() que RolloutAgent, así que
    `collect_episode` lo acepta sin cambios."""

    def __init__(self, num_actions: int):
        self.num_actions = num_actions

    def reset(self):
        pass

    def act(self, obs: dict, greedy: bool = False) -> int:
        return random.randrange(self.num_actions)


def collect_episode(env, agent, buffer, config: dict, num_actions: int, greedy: bool = False) -> int:
    """Recoge un episodio completo de `env` y lo guarda en `buffer` (usado por ambos trainers).

    También guarda el step terminal `o_T`, en el que ya no se actúa, pero que el world model
    necesita ver para aprender a reconocer un estado terminal.

    Args:
        env: entorno con interfaz `reset() -> obs` y `step(action) -> (obs, reward, done, info)`.
            `info` puede llevar `"truncated"` (corte por `max_steps`).
        agent: agente con interfaz `reset()` y `act(obs, greedy)` (RolloutAgent o RandomAgent).
        buffer: SequenceReplayBuffer donde se guarda el episodio (add_step + end_episode).
        config: configuración completa (la necesita obs_to_step).
        num_actions: número de acciones discretas del espacio de acción.
        greedy: se pasa False al recoger datos de entrenamiento, para que el agente explore.

    Returns:
        Número de pasos en los que se actuó (sin contar el step terminal o_T). Con él se calcula,
        según `train_ratio`, cuántas actualizaciones de gradiente corresponden al episodio.
    """
    observation = env.reset()
    agent.reset()
    done = False
    prev_action = None      # a_{t-1}: todavía no se ha dado ningún paso, así que no hay acción previa
    prev_reward = 0.0       # r_{t-1}: tampoco hay recompensa antes del primer paso
    info = {}
    length = 0

    # Dentro del bucle `observation` nunca es terminal (si lo fuera, el bucle habría acabado en
    # la iteración anterior), así que cont=1.0 es correcto para todos estos steps.
    while not done:
        action = agent.act(observation, greedy=greedy)
        next_obs, reward, done, info = env.step(action)
        length += 1

        step_data = obs_to_step(observation, config)
        step_data.update(prev_action=action_to_onehot(prev_action, num_actions),
                          reward=prev_reward, cont=1.0)
        buffer.add_step(step_data)

        observation, prev_action, prev_reward = next_obs, action, reward

    # Step terminal o_T: cont=1.0 si el episodio acaba por truncamiento (no es un final real) y
    # cont=0.0 si es un final real.
    cont_final = 1.0 if info.get("truncated") else 0.0
    step_data = obs_to_step(observation, config)
    step_data.update(prev_action=action_to_onehot(prev_action, num_actions),
                      reward=prev_reward, cont=cont_final)
    buffer.add_step(step_data)

    buffer.end_episode()
    return length


def flatten_states(states: dict) -> dict:
    """Aplana [B, T, dim] → [B*T, dim] y desconecta del grafo de autodiferenciación.

    Permite usar cualquier estado observado como punto de partida de rssm.imagine() sin
    que los gradientes de la imaginación se propaguen de vuelta al paso de observación.
    """
    return {k: v.reshape(-1, v.shape[-1]).detach() for k, v in states.items()}
