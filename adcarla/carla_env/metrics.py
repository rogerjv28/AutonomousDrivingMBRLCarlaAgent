"""DrivingMetrics: acumula colisiones, infracciones, progreso y éxito para la evaluación."""

from dataclasses import dataclass

@dataclass
class EpisodeStats:
    """Contadores acumulados de un único episodio."""
    steps: int = 0
    collisions: int = 0
    infractions: int = 0
    route_completion: float = 0.0   # De 0 a 1
    success: bool = False

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
            info: dict con "collision" e "infraction" (booleanos), como el que devuelve RewardFunction.
            route_completion: fracción de ruta completada acumulada hasta este step, se guarda el
                máximo visto en el episodio.
        """
        self._current_episode.steps += 1
        if info.get("collision"):
            self._current_episode.collisions += 1
        if info.get("infraction"):
            self._current_episode.infractions += 1
        self._current_episode.route_completion = max(self._current_episode.route_completion, float(route_completion))

    def end_episode(self, success: bool):
        """Cierra el episodio en curso: evalúa si tuvo éxito y lo guarda en self.episodes."""
        self._current_episode.success = bool(success)
        self.episodes.append(self._current_episode)

    def summary(self) -> dict:
        """Medias agregadas sobre todos los episodios cerrados (ratio de éxito, colisiones/episodio,
        infracciones/episodio, media de finalización)."""
        n = len(self.episodes) or 1
        return {
            "episodes": len(self.episodes),
            "success_rate": sum(episode.success for episode in self.episodes) / n,
            "collisions_per_episode": sum(episode.collisions for episode in self.episodes) / n,
            "infractions_per_episode": sum(episode.infractions for episode in self.episodes) / n,
            "mean_route_completion": sum(episode.route_completion for episode in self.episodes) / n,
        }
