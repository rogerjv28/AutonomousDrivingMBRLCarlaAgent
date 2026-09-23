"""TrafficSpawner: tráfico de fondo (vehículos + peatones) para poblar el mundo.

Vehículos: autopilot vía TrafficManager. Peatones: spawn del walker +
`controller.ai.walker`. Racional del sorteo determinista y del reseeding por episodio.
"""
import logging
import random

logger = logging.getLogger(__name__)


class TrafficSpawner:
    """Crea y destruye el tráfico de fondo (NPCs) de cada episodio."""

    def __init__(self, config: dict, client=None):
        """Lee el bloque `traffic` del config y, si hay cliente, conecta el TrafficManager.

        Args:
            config: configuración completa, se lee el bloque "traffic" y la semilla global.
            client: carla.Client ya conectado, si es None no se toca el TrafficManager.
        """
        traffic_config = config.get("traffic", {})
        self.num_vehicles = int(traffic_config.get("num_vehicles", 0))
        self.num_walkers = int(traffic_config.get("num_walkers", 0))
        self.tm_port = int(traffic_config.get("tm_port", 8000))
        self.seed = int(traffic_config.get("seed", config.get("seed", 0)))
        self._episode = 0
        self.episode_seed = self.seed
        self._rng = random.Random(self.seed)

        self.tm = None
        if client is not None:
            self.tm = client.get_trafficmanager(self.tm_port)
            self.tm.set_synchronous_mode(True)
            self.tm.set_random_device_seed(self.seed)

        self._vehicles = []
        self._walkers = []
        self._controllers = []

    def spawn(self, world, carla=None):
        """Puebla `world` con `num_vehicles` vehículos y `num_walkers` peatones.

        No limpia tráfico anterior: `CarlaEnv` llama a `destroy_all()` en `_cleanup()` antes.

        Args:
            world: carla.World donde se crean los actores.
            carla: módulo `carla` ya importado, si es None se importa aquí.
        """
        if carla is None:
            import carla
        # Seed determinista por episodio: misma semilla y mismo episodio -> mismo tráfico,
        # pase lo que pase con los spawns fallidos del episodio anterior.
        self.episode_seed = self.seed * 1000003 + self._episode
        self._rng = random.Random(self.episode_seed)
        self._episode += 1
        if self.tm is not None:
            self.tm.set_random_device_seed(self.episode_seed)   # el TM lleva su propio RNG
        if self.num_vehicles:
            self._spawn_vehicles(world)
        if self.num_walkers:
            self._spawn_walkers(world, carla)

    def _spawn_vehicles(self, world) -> list:
        """Autopilot vía TrafficManager sobre spawn points barajados con el RNG propio.

        Registra cada actor en `self._vehicles` en cuanto se crea, mismo patrón que
        `_spawn_walkers`.
        """
        blueprints = world.get_blueprint_library().filter("vehicle.*")
        spawn_points = list(world.get_map().get_spawn_points())
        self._rng.shuffle(spawn_points)

        for transform in spawn_points:
            if len(self._vehicles) >= self.num_vehicles:
                break
            blueprint = blueprints[self._rng.randrange(len(blueprints))]
            actor = world.try_spawn_actor(blueprint, transform)
            if actor is not None:   # punto ocupado (p.ej. por el ego): se descarta, no se reintenta
                actor.set_autopilot(True, self.tm_port)
                self._vehicles.append(actor)

        if len(self._vehicles) < self.num_vehicles:
            logger.warning("TrafficSpawner: se pidieron %d vehiculos y solo se han podido crear %d "
                            "(spawn points ocupados o insuficientes)", self.num_vehicles, len(self._vehicles))
        return self._vehicles

    def _spawn_walkers(self, world, carla) -> tuple:
        """Spawnea peatones y les adjunta un `WalkerAIController` que los manda a caminar.

        Registra cada walker y cada controlador en cuanto se crea, no al final del método.
        El fallo de creación de un controlador se recupera en el sitio, destruyendo de inmediato lo ya creado.
        """
        walker_blueprints = world.get_blueprint_library().filter("walker.pedestrian.*")
        world.set_pedestrians_seed(self.episode_seed)  # la malla la sortea el servidor, no self._rng

        for _ in range(self.num_walkers):
            location = world.get_random_location_from_navigation()
            if location is None:   # la malla de navegación no siempre da un punto válido
                continue
            blueprint = walker_blueprints[self._rng.randrange(len(walker_blueprints))]
            actor = world.try_spawn_actor(blueprint, carla.Transform(location))
            if actor is not None:
                self._walkers.append(actor)
        world.tick()

        controller_bp = world.get_blueprint_library().find("controller.ai.walker")
        try:
            for walker in self._walkers:
                self._controllers.append(world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker))
        except Exception:
            logger.warning("TrafficSpawner: fallo creando un controlador de peaton a medio camino; "
                            "se destruyen los %d walkers y %d controladores ya creados",
                            len(self._walkers), len(self._controllers))
            for actor in self._controllers + self._walkers:
                try:
                    actor.destroy()
                except Exception:
                    pass
            self._walkers, self._controllers = [], []
            return self._walkers, self._controllers
        world.tick()

        for controller in self._controllers:
            controller.start()
            # Mismo guard que arriba: sin punto válido no se asigna una posicion, el
            # controlador se queda arrancado sin destino inicial en vez de que lance
            # go_to_location(None). La llamada real de CARLA no admite None.
            destino = world.get_random_location_from_navigation()
            if destino is not None:
                controller.go_to_location(destino)

        return self._walkers, self._controllers

    def destroy_all(self):
        """Detiene y destruye todos los NPCs vivos (vehículos, peatones y sus controladores)."""
        for controller in self._controllers:
            try:
                controller.stop()
            except Exception:
                pass
        for actor in self._controllers + self._walkers + self._vehicles:
            try:
                actor.destroy()
            except Exception:
                pass
        self._vehicles, self._walkers, self._controllers = [], [], []

    def shutdown(self):
        """Saca al TrafficManager del modo síncrono. Llamar solo al cerrar el env (no en cada reset):
        dejarlo síncrono con el mundo ya en modo asíncrono lo deja esperando ticks que no llegan."""
        if self.tm is not None:
            self.tm.set_synchronous_mode(False)

    @property
    def actors(self) -> list:
        """Todos los actores NPC vivos (para tests y logging)."""
        return self._vehicles + self._walkers + self._controllers
