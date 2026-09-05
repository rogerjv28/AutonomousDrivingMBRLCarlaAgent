import math
import numpy as np

from .sensors import SensorSuite
from .actions import DiscreteActionSpace
from .reward import RewardFunction
from .metrics import DrivingMetrics
from .bev_privileged import PrivilegedBEVGenerator
from .route_tracker import RouteTracker, NUM_COMMANDS
from .scenarios import RouteManager
from .traffic import TrafficSpawner


def _offset_z(transform, delta: float):
    """Copia un `carla.Transform` elevando su z (para reintentar un spawn ocupado)."""
    copia = type(transform)(transform.location, transform.rotation)
    copia.location.z = transform.location.z + delta
    return copia


def _validate_terminal_block(blocked_steps_limit: int, max_steps: int):
    """Aborta el arranque si el terminal de "vehículo bloqueado" no se puede alcanzar.

    Un config incoherente (umbral de bloqueo más largo que el propio episodio) debe fallar
    aquí, no producir en silencio un experimento donde ese terminal y `weight_blocked` son
    código muerto.

    Raises:
        RuntimeError: si `blocked_steps_limit >= max_steps`.
    """
    if blocked_steps_limit >= max_steps:
        raise RuntimeError(
            f"El terminal de 'vehículo bloqueado' es inalcanzable: hacen falta {blocked_steps_limit} "
            f"pasos consecutivos por debajo de 0.1 m/s (reward.blocked_seconds / "
            f"carla.fixed_delta_seconds), pero la ruta más corta trunca el episodio a los "
            f"{max_steps} pasos (scenarios.max_steps). Baja 'reward.blocked_seconds' o sube "
            f"'scenarios.max_steps' en el config.")


class CarlaEnv:
    """Entorno CARLA con API tipo Gym (reset/step/close)."""

    def __init__(self, config: dict, route_manager = None, split: str = "train"):
        """Conecta con el servidor CARLA, carga el mundo/mapa y activa el modo síncrono.

        Args:
            config: diccionario de configuración (ver configs/), con las claves "carla",
            "scenarios" y "traffic".
            route_manager: RouteManager ya construido, si es None se crea con el bloque
                "scenarios" del config y los spawn points del mapa cargado.
            split: split de rutas que recorren los episodios ("train" o "eval").
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
        self.tracker = None         # RouteTracker del episodio en curso
        self.route_spec = None      # RouteSpec del episodio en curso
        self._route = []            # waypoints (RouteWaypoint) de la ruta trazada
        self._last_speed = 0.0
        self._steps = 0

        # Rutas fijas y reproducibles: las dos ramas recorren los mismos trayectos en el mismo
        # orden (ver scenarios.py). El planner se crea una vez y se reutiliza entre episodios.
        self.route_manager = route_manager or RouteManager(config, len(self.map.get_spawn_points()))
        self.traffic = TrafficSpawner(config, self.client)
        self.split = split
        self._episode = 0
        self._planner = None

        scenarios_config = config.get("scenarios", {})
        self._lookahead_waypoints = int(scenarios_config.get("lookahead_waypoints", 20))
        self._target_distance = float(scenarios_config.get("target_distance", 10.0))

        # Terminaciones por desviacion y por bloqueo (Think2Drive Sec. 6.4)
        reward_config = config.get("reward", {})
        self._max_deviation = float(reward_config.get("max_deviation_meters", 3.5))
        self._blocked_steps_limit = int(float(reward_config.get("blocked_seconds", 50.0)) / self.delta_time)
        self._blocked_steps = 0
        self._last_steer = 0.0      # para c_steer (Think2Drive): coste al cambiar de steer
        self._was_at_junction = False   # para detectar el instante de ENTRAR en un cruce

        # Guard de arranque: con la config por defecto la ruta mas corta debe truncar
        # despues de que el bloqueo sea alcanzable, o el terminal seria codigo muerto.
        total_route_specs = self.route_manager.routes("train") + self.route_manager.routes("eval")
        min_max_steps = min((spec.max_steps for spec in total_route_specs),
                            default=int(scenarios_config.get("max_steps", 600)))
        _validate_terminal_block(self._blocked_steps_limit, min_max_steps)

    def reset(self, split: str = None, weather: str = None):
        """Reinicia el episodio con la ruta que toca y estabiliza los sensores.

        Args:
            split: split de rutas para este episodio, por defecto el del constructor.
            weather: preset de clima que pisa el de la ruta. Lo usa la evaluación fuera de
                distribución, que recorre las mismas rutas bajo varios climas.

        Returns:
            La primera observación del episodio (ver _build_obs).
        """
        self._cleanup()
        self.world.tick()   # sincrono: destroy() no surte efecto hasta el siguiente tick

        # Ruta del episodio: round-robin determinista sobre el split (ver RouteManager.select).
        spec = self.route_manager.select(self._episode, split or self.split)
        self.route_spec = spec
        self._episode += 1
        self.route_manager.apply_weather(self.world, spec, weather=weather)

        # Spawn del ego en el origen de la ruta y trazado con el GlobalRoutePlanner.
        vehicle_blueprint = self.world.get_blueprint_library().find("vehicle.lincoln.mkz_2020")
        spawn = self.route_manager.spawn_transform(self.map, spec)
        self.ego = self._spawn_ego(vehicle_blueprint, spawn)
        self.traffic.spawn(self.world)  # tráfico de fondo del episodio
        if self._planner is None:
            self._planner = self.route_manager.build_planner(self.map)
        self._route = self.route_manager.build_route(self.map, spec, planner=self._planner)
        self.tracker = RouteTracker(self._route, lookahead_waypoints=self._lookahead_waypoints,
                                    target_distance=self._target_distance)

        # Variables iniciales
        self._last_speed = 0.0
        self._blocked_steps = 0
        self._steps = 0
        self._last_steer = 0.0  # para c_steer (Think2Drive): coste al cambiar de steer
        self._was_at_junction = False   # para detectar el instante de entrar en un cruce

        # Sensores y métricas
        self.sensors = SensorSuite(self.world, self.ego, self.config)
        self.metrics.reset_episode()

        for _ in range(5):  # avanza 5 pasos para recibir información veraz de los sensores
            self.world.tick()
        self.sensors.clear_queues()  # descarta las lecturas de calentamiento (sin frame de referencia fiable)

        self.world.tick()  # tick real: su frame es el que se empareja en _build_obs()
        self._update_tracker()   # tras el calentamiento: el ego ya esta asentado en el spawn
        self.sensors.pop_events()   # descarta colisiones/invasiones provocadas por el propio spawn

        return self._build_obs()

    def step(self, action: int):
        """Aplica una acción discreta, avanza un tick de simulación y calcula reward/done/info.

        Args:
            action: índice de acción discreta (ver DiscreteActionSpace).

        Returns:
            Tupla (obs, reward, done, info) estilo Gym; info incluye "route_completion",
            "deviation_meters" y "truncated" (corte por RouteSpec.max_steps).
        """
        # Realiza la acción, guardando el steer para el coste c_steer de Think2Drive
        control = self.actions.to_control(action)
        steer_changed = control.steer != self._last_steer
        self._last_steer = control.steer
        self.ego.apply_control(self.actions.to_carla(action))
        self.world.tick()
        self._steps += 1

        # Actualización variables
        events = self.sensors.pop_events()
        progress = self._update_tracker()
        impact_speed = self._last_speed   # velocidad de llegada al impacto: el tick ya ha frenado el coche
        speed = self._speed()
        self._last_speed = speed

        # Bloqueo: el coche lleva demasiado tiempo parado.
        self._blocked_steps = self._blocked_steps + 1 if speed < 0.1 else 0
        blocked = self._blocked_steps >= self._blocked_steps_limit

        signals = {"progress_meters": progress, "speed": speed, "impact_speed": impact_speed,
                   "collision": events["collision"],
                   "infraction": events["lane_invasion"], "steer_changed": steer_changed,
                   "route_done": self.tracker.route_done, "deviation_meters": self.tracker.deviation,
                   "max_deviation_meters": self._max_deviation, "blocked": blocked,
                   "red_light_violation": self._red_light_violation()}
        reward, done, info = self.reward_fn(signals)

        # Corte por tiempo: sin esto un coche dando vueltas no termina nunca el episodio.
        truncated = self._steps >= int(self.route_spec.max_steps)
        done = done or truncated

        completion = self.tracker.completion
        # El tipo del actor impactado no es parte de la recompensa (que solo mira si hubo choque),
        # pero sí del Driving Score: va del sensor a las métricas sin pasar por reward_fn.
        self.metrics.update({**info, "collision_type": events.get("collision_type")},
                            route_completion=completion)

        if done:
            self.metrics.end_episode(success=self.tracker.route_done and not events["collision"])

        return self._build_obs(), reward, done, {**info, "route_completion": completion,
                                                 "deviation_meters": self.tracker.deviation,
                                                 "truncated": truncated}

    # ---- HELPERS ----
    def _speed(self) -> float:
        """Velocidad actual del ego en m/s (módulo del vector de velocidad)."""
        v = self.ego.get_velocity()
        return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)

    def _red_light_violation(self) -> bool:
        """Detecta el instante de entrar en un cruce con el semáforo en rojo (Think2Drive Sec. 6.3).

        Evento puntual (edge-detection sobre `is_junction`), no una condición continua: si no se
        mirase la transición, un coche parado dentro del cruce en rojo penalizaría en cada tick.
        """
        waypoint = self.map.get_waypoint(self.ego.get_transform().location)
        at_junction = bool(waypoint and waypoint.is_junction)
        entering = at_junction and not self._was_at_junction
        self._was_at_junction = at_junction
        if not entering:
            return False

        traffic_light_state = self.ego.get_traffic_light_state()
        return "Red" in str(traffic_light_state)

    def _update_tracker(self) -> float:
        """Pasa la pose del ego al RouteTracker y devuelve los metros de progreso del paso."""
        transform = self.ego.get_transform()

        return self.tracker.update(transform.location.x, transform.location.y,
                                   transform.rotation.yaw)

    def _build_obs(self) -> dict:
        """Construye la observación: salidas de los sensores + estado + BEV privilegiado opcional.

        Empareja los sensores por el frame del snapshot actual (world.get_snapshot().frame), no
        por lo último que haya llegado: en modo síncrono el callback de un sensor puede completarse
        después de que world.tick() devuelva, y usar la última lectura cacheada mezclaría en una
        misma observación datos de dos frames simulados distintos (carrera sensor/tick).
        """
        # Obten observacion de sensores, emparejada exactamente con el frame actual
        if self.sensors:
            frame = self.world.get_snapshot().frame
            obs = self.sensors.get_frame(frame, timeout=2.0)
        else:
            obs = {}
        speed = self._speed()
        obs["state"] = np.array([speed], dtype=np.float32)

        # Guía de navegación que consume la política: target point en el frame del ego y
        # comando de alto nivel.
        obs["target_point"] = self.tracker.target_point
        obs["command"] = self.tracker.command

        # Medidas del ego para la política: sin esto el agente no sabe a qué velocidad va
        # ni adónde debe ir, y en un cruce le es físicamente imposible acertar.
        onehot_command = np.zeros(NUM_COMMANDS, dtype=np.float32)
        onehot_command[int(obs["command"])] = 1.0
        obs["measurements"] = np.concatenate([
            np.array([speed / 8.0, obs["target_point"][0] / 50.0, obs["target_point"][1] / 50.0],
                     dtype=np.float32),
            onehot_command,
        ])

        # Genera BEV privilegiado (si necesario)
        if self.use_privileged_bev and self.ego is not None:
            obs["bev_privileged"] = self.bev_generator.generate(self.world, self.ego, self._route)
        else:
            obs["bev_privileged"] = None
        
        return obs

    def _spawn_ego(self, blueprint, spawn, attempts: int = 4):
        """Spawnea el ego reintentando con un pequeno offset en z si el punto esta ocupado.

        Con rutas fijas en round-robin el mismo punto se reutiliza cada pocos episodios y basta
        con que un NPC haya quedado cerca para que spawn_actor lance "collision at spawn position".
        """
        transform = spawn
        for attempt in range(attempts):
            actor = self.world.try_spawn_actor(blueprint, transform)
            if actor is not None:
                return actor
            transform = _offset_z(transform, 0.5 * (attempt + 1))
            self.world.tick()
        raise RuntimeError(f"No se pudo spawnear el ego en {spawn.location} tras {attempts} intentos")

    def _cleanup(self):
        """Destruye sensores, ego y tráfico de fondo del episodio anterior, si existen."""
        if self.sensors:
            self.sensors.destroy(); self.sensors = None
        if self.ego:
            try:
                self.ego.destroy()
            except Exception:
                pass
            self.ego = None
        self.traffic.destroy_all()

    def close(self):
        """Limpia los actores y desactiva el modo síncrono del mundo CARLA y del TrafficManager."""
        self._cleanup()
        self.traffic.shutdown()
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)
