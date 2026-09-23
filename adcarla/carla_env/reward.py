"""RewardFunction: recompensa densa de Think2Drive Sec. 6.4, adoptada tal cual por Raw2Drive (Ap. C.4).

Función casi pura: recibe un dict de señales ya extraídas del simulador y devuelve
(reward, done, info). Así es testeable sin CARLA.

Correspondencia término a término con Think2Drive/Roach y el racional del rebalanceo de pesos.
"""

class RewardFunction:
    """Combina progreso, desviación lateral, velocidad, confort y eventos en una recompensa densa."""

    def __init__(self, cfg: dict):
        """Lee los pesos de cada término desde cfg["reward"] (ver configs/base.yaml).

        Args:
            cfg: configuración completa; se usa la clave "reward" y, para el paso de
                simulación, "carla.fixed_delta_seconds".
        """
        reward_config = cfg.get("reward", {})
        self.weight_progress = reward_config.get("weight_progress", 1.0)
        self.weight_collision = reward_config.get("weight_collision", -1.0)
        self.weight_infraction = reward_config.get("weight_infraction", -1.0)
        # c_steer de Think2Drive penaliza cambios de dirección del volante (alpha_st = 0.5).
        # Roach hace lo mismo con r_action = -0.1. Con acciones discretas (13 angulos fijos)
        # "distinto" es igualdad exacta, sin umbral.
        self.weight_steer = reward_config.get("weight_steer", -0.5)

        # Think2Drive pondera la desviación al doble que el progreso (α_de = 2.0 vs α_tr = 1.0).
        self.weight_deviation = reward_config.get("weight_deviation", -2.0)

        # Terminaciones estilo Roach: salirse de la ruta y quedarse bloqueado.
        self.weight_route_deviation = reward_config.get("weight_route_deviation", -1.0)
        self.weight_blocked = reward_config.get("weight_blocked", -1.0)

        # Progreso normalizado por la distancia máxima recorrible en un tick: sin esto,
        # avanzar rápido acumula más puntos por segundo cuantos más ticks corran, sin límite.
        self.v_max = float(reward_config.get("v_max", 8.0))
        dt = float(cfg.get("carla", {}).get("fixed_delta_seconds", 0.1))
        self._progress_scale = self.v_max * dt

        # r_speed (Think2Drive, simplificado): 0 por debajo de v_max, proporcional al exceso si no.
        self.weight_speed = reward_config.get("weight_speed", -0.1)

        # Semáforo en rojo (Think2Drive Sec. 6.3, condición 5): penaliza por evento, no termina
        # el episodio (Roach sí lo termina).
        self.weight_red_light = reward_config.get("weight_red_light", -2.0)

        # Colisión proporcional a la velocidad de impacto (Roach Ap. C.1), con
        # tope para que un choque a gran velocidad no desborde el resto de la recompensa.
        self.collision_speed_cap = float(reward_config.get("collision_speed_cap", 10.0))

    def __call__(self, signals: dict):
        """Calcula reward/done/info a partir de señales ya extraídas del simulador.

        Args:
            signals: dict con progress_meters (float), speed (float, m/s), collision (bool),
                impact_speed (float, m/s, velocidad del tick anterior al choque, por defecto speed),
                infraction (bool, solo marcas continuas), steer_changed (bool), route_done (bool),
                deviation_meters (float), max_deviation_meters (float), blocked (bool),
                red_light_violation (bool).

        Returns:
            Tupla (reward, done, info). info incluye collision, infraction, route_done,
            route_deviation, blocked, red_light_violation y deviation_norm.
        """
        progress = float(signals.get("progress_meters", 0.0))
        speed = float(signals.get("speed", 0.0))
        # Roach penaliza con la velocidad de LLEGADA al impacto, el tick del choque ya ha frenado
        # el coche, asi que `speed` subestima el golpe. El entorno pasa la del tick anterior.
        impact_speed = float(signals.get("impact_speed", speed))
        collision = bool(signals.get("collision", False))
        infraction = bool(signals.get("infraction", False))
        steer_changed = bool(signals.get("steer_changed", False))
        route_done = bool(signals.get("route_done", False))
        blocked = bool(signals.get("blocked", False))
        red_light_violation = bool(signals.get("red_light_violation", False))

        # Progreso por tick en [.., 1]: metros avanzados / metros máximos recorribles en un tick,
        # recortado por arriba a 1.0 (el recorte es la pieza clave del rebalanceo).
        # Puede bajar de 0 (retroceder resta).
        progress_norm = progress / self._progress_scale if self._progress_scale > 0 else progress
        progress_norm = min(progress_norm, 1.0)

        # Desviación lateral normalizada por el umbral máximo (Think2Drive: "normalized by the
        # max deviation threshold D_max"). Queda en [0, 1] para que el peso sea interpretable.
        max_deviation = float(signals.get("max_deviation_meters", 3.5))
        deviation = float(signals.get("deviation_meters", 0.0))
        deviation_norm = min(deviation / max_deviation, 1.0) if max_deviation > 0 else 0.0

        # Salirse de la ruta termina el episodio (Roach: Δ_p > 3.5 m -> terminal reward -1).
        route_deviation = deviation > max_deviation

        # r_speed: 0 si no se excede v_max, proporcional al exceso si no (estilo Roach, sin detector
        # de obstáculos).
        excess_speed = max(0.0, speed - self.v_max)

        # Reward Function
        reward = (self.weight_progress * progress_norm
                  + self.weight_deviation * deviation_norm
                  + self.weight_speed * excess_speed
                  + (self.weight_steer if steer_changed else 0.0))
        if infraction:
            reward += self.weight_infraction
        if red_light_violation:
            reward += self.weight_red_light
        if route_deviation:
            reward += self.weight_route_deviation
        if blocked:
            reward += self.weight_blocked
        if collision:
            # Terminal -(1+min(s, tope)) de Roach: cuanto más rápido el impacto, más penaliza.
            reward += self.weight_collision * (1.0 + min(impact_speed, self.collision_speed_cap))

        done = collision or route_done or route_deviation or blocked
        info = {"collision": collision, "infraction": infraction, "route_done": route_done,
                "route_deviation": route_deviation, "blocked": blocked,
                "red_light_violation": red_light_violation, "deviation_norm": deviation_norm}

        return reward, done, info
