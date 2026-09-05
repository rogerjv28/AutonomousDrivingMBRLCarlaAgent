"""DrivingMetrics: colisiones, infracciones, progreso, éxito y Driving Score parecido al de CARLA Leaderboard.

DS = Route Completion x Infraction Penalty, con IP multiplicativo.
"""

from dataclasses import asdict, dataclass, field

# Coeficientes del Infraction Penalty del CARLA Leaderboard. No se inventa ninguno.
PENALTY_COEFICIENTS = {
    "pedestrian": 0.50,
    "vehicle": 0.60,
    "static": 0.65,
    "red_light": 0.70,
}

COLLISION_TYPES = ("pedestrian", "vehicle", "static")

# Aviso que viaja con los números (summary.json, tabla impresa, README): sin él es fácil comparar
# este DS con los publicados del Leaderboard como si fueran la misma métrica.
NOTA_DRIVING_SCORE = (
    "Driving Score = Route Completion x Infraction Penalty con los coeficientes del CARLA "
    "Leaderboard (peatón 0.50, vehículo 0.60, estático 0.65, semáforo en rojo 0.70). Es una cota "
    "superior del DS oficial: no se penalizan stops ignorado (0.80), sentido contrario, pisar "
    "arcén/acera ni timeout, y la invasión de línea continua se cuenta aparte pero no entra en"
    "el IP (el Leaderboard no le da coeficiente). Una colisión sin clasificar cuenta como estática."
    "No comparar con DS publicados sin declarar esta salvedad."
)


@dataclass
class EpisodeStats:
    """Contadores acumulados de un único episodio, con su Driving Score."""
    steps: int = 0
    collisions: int = 0
    collisions_by_type: dict = field(default_factory=lambda: {t: 0 for t in COLLISION_TYPES})
    red_lights: int = 0
    infractions: int = 0            # línea continua cruzada; fuera del IP (ver cabecera)
    route_completion: float = 0.0   # De 0 a 1
    success: bool = False

    @property
    def infraction_penalty(self) -> float:
        """IP multiplicativo del Leaderboard: producto de los coeficientes de lo cometido."""
        penalty = PENALTY_COEFICIENTS["red_light"] ** self.red_lights
        for type in COLLISION_TYPES:
            penalty *= PENALTY_COEFICIENTS[type] ** self.collisions_by_type[type]
        return penalty

    @property
    def driving_score(self) -> float:
        """DS = Route Completion x Infraction Penalty."""
        return self.route_completion * self.infraction_penalty

    def as_dict(self) -> dict:
        """Fila plana para el CSV de evaluación (el desglose por tipo, en columnas propias)."""
        metrics_row = asdict(self)
        metrics_row.pop("collisions_by_type")
        metrics_row["success"] = int(self.success)
        for type in COLLISION_TYPES:
            metrics_row[f"collisions_{type}"] = self.collisions_by_type[type]
        metrics_row["infraction_penalty"] = self.infraction_penalty
        metrics_row["driving_score"] = self.driving_score
        return metrics_row

    # Acceso cómodo desde los tests y desde el agregador de scripts/evaluate.py.
    @property
    def collisions_pedestrian(self) -> int:
        return self.collisions_by_type["pedestrian"]

    @property
    def collisions_vehicle(self) -> int:
        return self.collisions_by_type["vehicle"]

    @property
    def collisions_static(self) -> int:
        return self.collisions_by_type["static"]


class DrivingMetrics:
    """Acumula EpisodeStats episodio a episodio y calcula medias agregadas."""

    def __init__(self):
        self.episodes = []
        self._current_episode = EpisodeStats()

    def reset_episode(self):
        """Descarta el episodio en curso y empieza a contar uno nuevo desde cero."""
        self._current_episode = EpisodeStats()

    def update(self, info: dict, route_completion: float = 0.0):
        """Registra un step del episodio en curso.

        Args:
            info: dict con "collision", "infraction" y "red_light_violation" (booleanos), como el
                que devuelve RewardFunction, más "collision_type" ("pedestrian"/"vehicle"/"static")
                si el sensor pudo clasificar el choque. Una colisión sin clasificar cuenta como
                estática (0.65, el más suave de los tres): mejor quedarse corto que inventarse una
                penalización mayor de la que se puede demostrar.
            route_completion: fracción de ruta completada acumulada hasta este step, se guarda el
                máximo visto en el episodio.
        """
        self._current_episode.steps += 1
        if info.get("collision"):
            self._current_episode.collisions += 1
            type = info.get("collision_type") or "static"
            if type not in COLLISION_TYPES:
                type = "static"
            self._current_episode.collisions_by_type[type] += 1
        if info.get("red_light_violation"):
            self._current_episode.red_lights += 1
        if info.get("infraction"):
            self._current_episode.infractions += 1
        self._current_episode.route_completion = max(self._current_episode.route_completion, float(route_completion))

    def end_episode(self, success: bool):
        """Cierra el episodio en curso: evalúa si tuvo éxito y lo guarda en self.episodes."""
        self._current_episode.success = bool(success)
        self.episodes.append(self._current_episode)

    def summary(self, window: int = None) -> dict:
        """Medias agregadas por episodio (ratio de éxito, Driving Score, colisiones por tipo,
        infracciones y finalización de ruta).

        Args:
            window: si se indica, promedia solo los últimos `window` episodios. La media
                histórica diluye la mejora reciente entre miles de valores viejos y deja la
                curva de aprendizaje plana, el entrenamiento la pide con ventana móvil.
        """
        episodios = self.episodes[-window:] if window else self.episodes
        n = len(episodios) or 1
        metric_summary = {
            "episodes": len(self.episodes),   # total cerrados, no el de la ventana
            "success_rate": sum(episode.success for episode in episodios) / n,
            "mean_driving_score": sum(episode.driving_score for episode in episodios) / n,
            "mean_infraction_penalty": sum(episode.infraction_penalty for episode in episodios) / n,
            "collisions_per_episode": sum(episode.collisions for episode in episodios) / n,
            "red_lights_per_episode": sum(episode.red_lights for episode in episodios) / n,
            "infractions_per_episode": sum(episode.infractions for episode in episodios) / n,
            "mean_route_completion": sum(episode.route_completion for episode in episodios) / n,
        }
        for type in COLLISION_TYPES:
            metric_summary[f"collisions_{type}_per_episode"] = \
                sum(episode.collisions_by_type[type] for episode in episodios) / n
        return metric_summary
