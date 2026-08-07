import math
import numpy as np

from .sensors import SensorSuite
from .actions import DiscreteActionSpace
from .reward import RewardFunction
from .metrics import DrivingMetrics
from .bev_privileged import PrivilegedBEVGenerator

def _dist(a, b) -> float:
    """Distancia euclídea 3D entre dos puntos con atributos x, y, z."""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

class CarlaEnv:
    """Entorno CARLA con API tipo Gym (reset/step/close)."""

    def __init__(self, config: dict):
        """Conecta con el servidor CARLA, carga el mundo/mapa y activa el modo síncrono.

        Args:
            config: diccionario de configuración (ver configs/), con la clave "carla"
                para host/port/timeout/town/fixed_delta_seconds/no_rendering.
        """
        import carla
        # Instanciamos todas las variables que necesitaremos y inicializamos la conexión con CARLA con la configuración que nos llega
        self.carla = carla
        self.config = config
        carla_config = config.get("carla", {})
        self.client = carla.Client(carla_config.get("host", "localhost"), carla_config.get("port", 2000))
        self.client.set_timeout(float(carla_config.get("timeout", 60.0)))

        # Ciudad
        town = carla_config.get("town", "Town01")

        # Mundo
        world = self.client.get_world()
        if not world.get_map().name.endswith(town):    # evita recargar el mapa
            world = self.client.load_world(town)
        self.world = world

        # Mapa
        self.map = self.world.get_map()

        # Frecuencia refresco (segundos)
        self.delta_time = float(carla_config.get("fixed_delta_seconds", 0.1))

        # Configuración
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.delta_time
        settings.no_rendering_mode = bool(carla_config.get("no_rendering", False))  # el profesor no necesita render
        self.world.apply_settings(settings)

        self.actions = DiscreteActionSpace()
        self.reward_fn = RewardFunction(config)
        self.metrics = DrivingMetrics()
        self.bev_generator = PrivilegedBEVGenerator(config)
        self.use_privileged_bev = bool(config.get("privileged_bev", True))

        self.ego = None
        self.sensors = None
        self._route = []    # lista de carla.Location
        self._cumulative_distance = []  # distancia acumulada por waypoint
        self._route_index = 0
        self._last_speed = 0.0
        self._route_len_waypoint = 200

    # Ruta simplificada
    def _build_route(self, start_waypoint):
        """Genera una ruta de waypoints cada 2 m a partir de start_waypoint y su distancia acumulada.

        Returns:
            Tupla (route, cumulative_dist): lista de carla.Location y lista de distancia acumulada
            (misma longitud) hasta cada punto de la ruta.
        """
        # Inizialización variables
        route, cumulative_dist, waypoint, total = [], [], start_waypoint, 0.0
        route.append(waypoint.transform.location)
        cumulative_dist.append(0.0)

        for _ in range(self._route_len_waypoint):
            # Cálculo waypoint siguiente
            next_waypoint = waypoint.next(2.0)
            if not next_waypoint:
                break
            previous_waypoint = waypoint.transform.location

            # Actualización variables
            waypoint = next_waypoint[0]
            total += _dist(previous_waypoint, waypoint.transform.location)
            route.append(waypoint.transform.location)
            cumulative_dist.append(total)

        return route, cumulative_dist

    def reset(self):
        """Reinicia el episodio: respawnea el ego, recalcula la ruta y estabiliza los sensores.

        Returns:
            La primera observación del episodio (ver _build_obs).
        """
        # Limpieza y spawn del vehiculo
        self._cleanup()
        vehicle_blueprint = self.world.get_blueprint_library().find("vehicle.lincoln.mkz_2020")
        spawn = np.random.choice(self.map.get_spawn_points())

        # Variables iniciales
        self.ego = self.world.spawn_actor(vehicle_blueprint, spawn)
        start_waypoint = self.map.get_waypoint(spawn.location)
        self._route, self._cumulative_distance = self._build_route(start_waypoint)
        self._route_index = 0
        self._last_speed = 0.0

        # Sensores y métricas
        self.sensors = SensorSuite(self.world, self.ego, self.config)
        self.metrics.reset_episode()

        for _ in range(5):  # avanza 5 pasos para recibir información veraz de los sensores
            self.world.tick()

        return self._build_obs()

    def step(self, action: int):
        """Aplica una acción discreta, avanza un tick de simulación y calcula reward/done/info.

        Args:
            action: índice de acción discreta (ver DiscreteActionSpace).

        Returns:
            Tupla (obs, reward, done, info) estilo Gym; info incluye "route_completion".
        """
        # Realiza la acción
        self.ego.apply_control(self.actions.to_carla(action))
        self.world.tick()

        # Actualización variables
        events = self.sensors.pop_events()
        progress = self._update_route_progress()
        speed = self._speed()
        jerk = (speed - self._last_speed) / self.delta_time
        self._last_speed = speed
        route_done = self._route_index >= len(self._route) - 2

        signals = {"progress_m": progress, "collision": events["collision"],
                   "infraction": events["lane_invasion"], "jerk": jerk, "route_done": route_done}
        reward, done, info = self.reward_fn(signals)

        completion = self._cumulative_distance[self._route_index] / (self._cumulative_distance[-1] + 1e-6)
        self.metrics.update(info, route_completion=completion)

        if done:
            self.metrics.end_episode(success=route_done and not events["collision"])

        return self._build_obs(), reward, done, {**info, "route_completion": completion}

    # ---- HELPERS ----
    def _speed(self) -> float:
        """Velocidad actual del ego en m/s (módulo del vector de velocidad)."""
        v = self.ego.get_velocity()
        return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)

    def _update_route_progress(self) -> float:
        """Avanza el índice de ruta si el ego ya está cerca del siguiente waypoint.

        Returns:
            Distancia recorrida a lo largo de la ruta desde la última llamada (metros).
        """
        loc = self.ego.get_transform().location
        old = self._cumulative_distance[self._route_index]
        while (self._route_index < len(self._route) - 1 and
               _dist(loc, self._route[self._route_index + 1]) < 3.0):
            self._route_index += 1
        return self._cumulative_distance[self._route_index] - old

    def _build_obs(self) -> dict:
        """Construye la observación: salidas de los sensores + estado + BEV privilegiado opcional."""
        # Obten observacion de sensores
        obs = self.sensors.get_obs() if self.sensors else {}
        obs["state"] = np.array([self._speed()], dtype=np.float32)

        # Genera BEV privilegiado (si necesario)
        if self.use_privileged_bev and self.ego is not None:
            obs["bev_privileged"] = self.bev_generator.generate(self.world, self.ego, self._route)
        else:
            obs["bev_privileged"] = None
        
        return obs

    def _cleanup(self):
        """Destruye sensores y ego del episodio anterior, si existen."""
        if self.sensors:
            self.sensors.destroy(); self.sensors = None
        if self.ego:
            try:
                self.ego.destroy()
            except Exception:
                pass
            self.ego = None

    def close(self):
        """Limpia los actores y desactiva el modo síncrono del mundo CARLA."""
        self._cleanup()
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)
