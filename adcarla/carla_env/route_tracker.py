"""RouteTracker: seguimiento de la ruta (numpy, sin API de CARLA).

Convierte la pose del ego y la ruta trazada en las señales que consumen la recompensa y la
política: progreso en metros, desviación lateral respecto al centro del carril, waypoint
actual, target point a 10 m en el frame del ego y comando de navegación vigente.

Interfaz que consume la política:

    tracker.update(x, y, yaw_deg)   # metros avanzados desde la llamada anterior
    tracker.target_point            # (x, y) en metros, en el frame del ego
    tracker.command                 # entero en [0, NUM_COMMANDS) -> onehot(6)

Convenio de ejes (el mismo que `bev_privileged.world_to_bev`): x del ego = longitudinal
(delante positivo), y del ego = lateral (derecha positiva), yaw en grados como en CARLA.
"""
from __future__ import annotations

import math
from enum import IntEnum
from typing import NamedTuple, Sequence

import numpy as np


class NavigationCommand(IntEnum):
    """Comando de navegación de alto nivel, en [0, 6) para el onehot de las medidas del ego.

    Son las seis `RoadOption` no nulas de `agents.navigation.local_planner`, renumeradas de
    1..6 a 0..5 (`VOID` no es una maniobra y cae a LANEFOLLOW).
    """
    LEFT = 0
    RIGHT = 1
    STRAIGHT = 2
    LANEFOLLOW = 3
    CHANGELANELEFT = 4
    CHANGELANERIGHT = 5


NUM_COMMANDS = len(NavigationCommand)


def command_from_road_option(road_option) -> int:
    """Traduce una `RoadOption` de CARLA (1..6, VOID = -1) al comando de este módulo (0..5).

    No importa `agents.navigation`: basta con el atributo `.value` del enum, así que el módulo
    sigue siendo puro y testeable sin CARLA.
    """
    value = getattr(road_option, "value", road_option)
    try:
        return int(NavigationCommand(int(value) - 1))
    except (TypeError, ValueError):
        return int(NavigationCommand.LANEFOLLOW)


class RouteWaypoint(NamedTuple):
    """Punto de la ruta ya desligado de CARLA: posición en el mundo y maniobra asociada.

    Es una tupla, así que `PrivilegedBEVGenerator` puede seguir leyendo `.x`/`.y` y los tests
    pueden pasar tuplas planas `(x, y, comando)`.
    """
    x: float
    y: float
    command: int


class RouteTracker:
    """Sigue la posición del ego a lo largo de una ruta de waypoints."""

    def __init__(self, waypoints: Sequence, lookahead_waypoints: int = 20,
                 target_distance: float = 10.0):
        """
        Args:
            waypoints: ruta como secuencia de `RouteWaypoint` o de tuplas (x, y, comando).
            lookahead_waypoints: ventana de búsqueda hacia delante al avanzar el índice.
            target_distance: distancia (m) a la que se toma el target point sobre la ruta.

        Raises:
            ValueError: si la ruta tiene menos de dos waypoints (el episodio acabaría al instante).
        """
        if len(waypoints) < 2:
            raise ValueError(f"La ruta necesita al menos 2 waypoints y tiene {len(waypoints)}.")

        self.waypoints = [RouteWaypoint(float(w[0]), float(w[1]), int(w[2])) for w in waypoints]
        self.lookahead_waypoints = int(lookahead_waypoints)
        self.target_distance = float(target_distance)

        points = np.array([(w.x, w.y) for w in self.waypoints], dtype=np.float64)
        sections = np.linalg.norm(np.diff(points, axis=0), axis=1)
        self._points = points
        self._cumulative = np.concatenate([[0.0], np.cumsum(sections)])

        self.index = 0
        self.deviation = 0.0
        self.target_point = np.zeros(2, dtype=np.float32)
        self.command = int(NavigationCommand.LANEFOLLOW)

    # ---- ESTADO DERIVADO ----
    @property
    def length_meters(self) -> float:
        """Longitud total de la ruta en metros."""
        return float(self._cumulative[-1])

    @property
    def completion(self) -> float:
        """Fracción de ruta recorrida, en [0, 1] (la métrica `route_completion` del TFM)."""
        return float(self._cumulative[self.index] / (self.length_meters + 1e-6))

    @property
    def route_done(self) -> bool:
        """True al alcanzar el final de la ruta (los dos últimos waypoints)."""
        return self.index >= len(self.waypoints) - 2

    # ---- ACTUALIZACIÓN ----
    def update(self, x: float, y: float, yaw_deg: float) -> float:
        """Actualiza el seguimiento con la pose del ego y devuelve el progreso del paso.

        Args:
            x, y: posición del ego en el mundo (metros).
            yaw_deg: orientación del ego en grados (convenio de CARLA).

        Returns:
            Metros avanzados a lo largo de la ruta desde la llamada anterior (nunca negativo).
        """
        previous = self._cumulative[self.index]
        self.index = self._nearest_index(x, y)
        self.deviation = self._lateral_deviation(x, y)
        self.target_point, self.command = self._target(x, y, yaw_deg)

        return float(self._cumulative[self.index] - previous)

    def _nearest_index(self, x: float, y: float) -> int:
        """Waypoint más cercano dentro de la ventana hacia delante; el índice nunca retrocede.
        """
        last = min(self.index + self.lookahead_waypoints, len(self.waypoints) - 1)
        window = self._points[self.index:last + 1]
        distances = np.linalg.norm(window - np.array([x, y]), axis=1)

        return self.index + int(np.argmin(distances))

    def _lateral_deviation(self, x: float, y: float) -> float:
        """Distancia perpendicular al tramo de ruta actual (r_position de Roach Ap. C.1).

        La ruta son waypoints del centro del carril, así que la distancia al segmento que une
        el waypoint actual con el siguiente es la desviación respecto al centro.
        """
        next = min(self.index + 1, len(self.waypoints) - 1)
        start, end = self._points[self.index], self._points[next]
        point = np.array([x, y]) - start
        segmento = end - start

        length_squared = float(segmento @ segmento)
        if length_squared < 1e-9:        # segmento degenerado (o final de ruta)
            return float(np.linalg.norm(point))

        # Proyección escalar acotada a [0, 1] para no salirse del segmento.
        t = min(1.0, max(0.0, float(point @ segmento) / length_squared))

        return float(np.linalg.norm(point - t * segmento))

    def _target(self, x: float, y: float, yaw_deg: float):
        """Target point (frame del ego) y comando del waypoint situado a distancia `target_distance` delante.

        El comando es el del waypoint objetivo, no el del pisado: lo que la política necesita
        saber es la maniobra que viene (girar, cambiar de carril), no la que está ejecutando.
        """
        objective_waypoint = int(np.searchsorted(self._cumulative,
                                       self._cumulative[self.index] + self.target_distance))
        objective_waypoint = min(objective_waypoint, len(self.waypoints) - 1)

        dx = self._points[objective_waypoint, 0] - x
        dy = self._points[objective_waypoint, 1] - y
        yaw = math.radians(float(yaw_deg))
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)

        target = np.array([dx * cos_yaw + dy * sin_yaw,          # longitudinal (delante)
                           -dx * sin_yaw + dy * cos_yaw],        # lateral (derecha)
                          dtype=np.float32)

        return target, int(self.waypoints[objective_waypoint].command)
