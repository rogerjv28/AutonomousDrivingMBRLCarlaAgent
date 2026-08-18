"""RewardFunction: recompensa densa (Think2Drive-style): progreso + colisión + infracción + confort.

Función casi pura: recibe un dict de señales ya extraídas del simulador y devuelve
(reward, done, info). Así es testeable sin CARLA.
"""

class RewardFunction:
    """Combina progreso, colisión, infracción y confort en una recompensa densa escalar."""

    def __init__(self, cfg: dict):
        """Lee los pesos de cada término desde cfg["reward"] (ver configs/base.yaml).

        Args:
            cfg: configuración completa; se usa la clave "reward" (weight_progress,
                weight_collision, weight_infraction, weight_comfort).
        """
        reward_config = cfg.get("reward", {})
        self.weight_progress = reward_config.get("weight_progress", 1.0)
        self.weight_collision = reward_config.get("weight_collision", -1.0)
        self.weight_infraction = reward_config.get("weight_infraction", -1.0)
        self.weight_comfort = reward_config.get("weight_comfort", -0.1)

    def __call__(self, signals: dict):
        """Calcula reward/done/info a partir de señales ya extraídas del simulador.

        Args:
            signals: dict con progress_meters (float), collision (bool), infraction (bool),
                jerk (float), route_done (bool).

        Returns:
            Tupla (reward, done, info), info incluye collision/infraction/route_done.
        """
        progress = float(signals.get("progress_meters", 0.0))
        collision = bool(signals.get("collision", False))
        infraction = bool(signals.get("infraction", False))
        jerk = float(signals.get("jerk", 0.0))
        route_done = bool(signals.get("route_done", False))

        # Reward Function
        reward = self.weight_progress * progress + self.weight_comfort * abs(jerk)
        if infraction:
            reward += self.weight_infraction
        if collision:
            reward += self.weight_collision

        done = collision or route_done
        info = {"collision": collision, "infraction": infraction, "route_done": route_done}
        
        return reward, done, info
