"""Utilidades de rollout: conversión de observaciones de CARLA y agente de conducción en lazo cerrado."""
import numpy as np
import torch
import torch.nn.functional as F
from adcarla.encoders.fusion.lidar_branch import LiDARBranch


def obs_to_step(obs: dict, config: dict) -> dict:
    """Convierte una observación raw de CarlaEnv en un dict numpy listo para el encoder y el decoder.

    Args:
        obs: dict de CarlaEnv (con claves "bev_privileged", "cameras", "lidar").
        config: dict de configuración completa.

    Returns:
        Dict de arrays numpy con las claves disponibles según la rama y los sensores activos.
    """
    step = {}

    # Máscara BEV privilegiada: solo disponible cuando privileged_bev=True en la configuración
    if obs.get("bev_privileged") is not None:
        step["bev"] = np.asarray(obs["bev_privileged"], dtype=np.float32)       # [Cb, Hb, Wb]

    # Cámaras: transponemos HWC→CHW (formato PyTorch) y normalizamos a [0, 1]
    cams = obs.get("cameras")
    if cams:
        cam_images = [np.transpose(c, (2, 0, 1)) for c in cams.values() if c is not None]
        if cam_images:
            step["cameras"] = (np.stack(cam_images, 0).astype(np.float32) / 255.0)    # [N, 3, H, W] (N cámaras)

    # LiDAR: solo rama fusión, rasteriza la nube de puntos a una proyección BEV en 2 canales
    if obs.get("lidar") is not None:
        lidar_range_m = float(config.get("bev", {}).get("range_meters", 50.0))
        step["lidar_bev"] = LiDARBranch.rasterize(np.asarray(obs["lidar"]), range_meters=lidar_range_m)  # [2, Hb, Wb]

    return step


def _batchify(step: dict, device) -> dict:
    """Añade dimensión de batch (B=1) a cada array del step y lo convierte en tensor float32 cargado en el device."""
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
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
            action_idx = int(torch.distributions.Categorical(logits=self.actor.net(feat)).sample())

        # Guarda la acción como one-hot para condicionar el siguiente obs_step del RSSM
        self.prev = F.one_hot(torch.tensor([action_idx], device=self.device), self.wm.num_actions).float()
        return action_idx


def flatten_states(states: dict) -> dict:
    """Aplana [B, T, dim] → [B*T, dim] y desconecta del grafo de autodiferenciación.

    Permite usar cualquier estado observado como punto de partida de rssm.imagine() sin
    que los gradientes de la imaginación se propaguen de vuelta al paso de observación.
    """
    return {k: v.reshape(-1, v.shape[-1]).detach() for k, v in states.items()}
