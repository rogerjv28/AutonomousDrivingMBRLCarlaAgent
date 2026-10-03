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
        self._world = None 

    def spawn(self, world, carla=None):
        """Puebla `world` con `num_vehicles` vehículos y `num_walkers` peatones.

        No limpia tráfico anterior: `CarlaEnv` llama a `destroy_all()` en `_cleanup()` antes.

        Args:
            world: carla.World donde se crean los actores.
            carla: módulo `carla` ya importado, si es None se importa aquí.
        """
        if carla is None:
            import carla
        self._world = world
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
            if actor is not None:   # punto ocupado (p. ej. por el ego): se descarta, no se reintenta
                self._vehicles.append(actor)   # Registrar antes de tocarlo: si set_autopilot peta, destroy_all() lo alcanza
                try:
                    actor.set_autopilot(True, self.tm_port)
                except Exception:
                    # Aqui no hay world.tick() entre spawn y esta llamada (el vehículo no ha tenido
                    # ocasion de "morir"), así que el riesgo es mínimo. El try/except solo mantiene el
                    # mismo patron defensivo que _spawn_walkers en todo sitio donde se toca un actor
                    # recien creado.
                    logger.warning("TrafficSpawner: fallo activando autopilot en un vehículo de fondo, se ignora")

        if len(self._vehicles) < self.num_vehicles:
            logger.warning("TrafficSpawner: se pidieron %d vehículos y solo se han podido crear %d "
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
            if actor is not None:   # None = punto de navegacion ocupado, try_spawn no reintenta
                self._walkers.append(actor)
        world.tick()

        controller_bp = world.get_blueprint_library().find("controller.ai.walker")
        try:
            for walker in self._walkers:
                if not walker.is_alive:
                    # Puede morir atropellado por el trafico de fondo (vehículos ya en autopilot) en
                    # el world.tick() de justo arriba, antes de que le toque su controlador. Sin este
                    # guard, spawn_actor(..., attach_to=walker) revienta con 'trying to operate on a
                    # destroyed actor' (SIGABRT nativo, no capturable) en vez de con una excepcion
                    # Python normal.
                    logger.warning("TrafficSpawner: un walker murió antes de asignarle controlador, se omite")
                    continue
                logger.debug("TrafficSpawner: spawn_actor(controller) attach_to walker id=%s",
                             getattr(walker, "id", "?"))
                self._controllers.append(world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker))
        except Exception:
            logger.warning("TrafficSpawner: fallo creando un controlador de peatón a medio camino; "
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
            try:
                # `controller.is_alive` responde por el controlador, no por su walker: entre el
                # world.tick() de arriba y este bucle el trafico de fondo puede haber atropellado a
                # un walker que ya tiene controlador adjunto pero aun sin arrancar. `controller.parent`
                # es ese walker. Operar sobre un controlador cuyo walker ya murió puede reproducir el SIGABRT nativo.
                logger.debug("TrafficSpawner: start() sobre controller id=%s (walker id=%s)",
                             getattr(controller, "id", "?"), getattr(controller.parent, "id", "?"))
                if controller.is_alive and (controller.parent is None or controller.parent.is_alive):
                    controller.start()
                    # Sin punto válido no se asigna posicion: el controlador se queda arrancado sin
                    # destino inicial en vez de llamar a go_to_location(None), que CARLA no admite.
                    destiny = world.get_random_location_from_navigation()
                    if destiny is not None:
                        controller.go_to_location(destiny)
                elif controller.is_alive:
                    logger.warning("TrafficSpawner: el walker de un controlador (id=%s) murió antes "
                                    "de arrancarlo; se omite", getattr(controller, "id", "?"))
            except Exception:
                # No es limpieza final sino un fallo durante el episodio activo: se deja rastro
                # en el log en vez de silenciarlo, pero un peatón no debe tumbar el reset entero.
                logger.warning("TrafficSpawner: fallo arrancando un controlador de peatón (id=%s); "
                                "se ignora y se continua", getattr(controller, "id", "?"))

        return self._walkers, self._controllers

    def destroy_all(self):
        """Detiene y destruye todos los NPCs vivos (vehículos, peatones y sus controladores).

        En tres fases, para no destruir un actor mientras un hilo en segundo plano del cliente de
        CARLA (el tick del TrafficManager, la IA de un walker) todavía lo referencia. Esa carrera
        es la que dispara el `std::terminate` / `trying to operate on a destroyed actor`.
        
        Etapas:

        1. Desengancharlo todo del bucle vivo: `set_autopilot(False)` saca los vehículos del
           TrafficManager; `controller.stop()` para la IA de los peatones.
        2. Un `world.tick()`: el servidor procesa esas bajas y los hilos en vuelo terminan antes
           de que se libere memoria.
        3. Entonces se hace el `destroy()`.

        Un walker puede además haber muerto atropellado a media escena: su controlador sigue "vivo"
        para CARLA aunque el walker no exista, así que el guard mira `controller.parent.is_alive`
        (el walker real). El `.destroy()` del controlador zombi se intenta igual.
        """
        # --- Fase 1: desenganchar del bucle vivo (sin destruir) ---
        for vehicle in self._vehicles:
            try:
                if vehicle.is_alive:
                    vehicle.set_autopilot(False, self.tm_port)
            except Exception:
                pass
        for controller in self._controllers:
            try:
                logger.debug("TrafficSpawner: stop() sobre controller id=%s (walker id=%s)",
                             getattr(controller, "id", "?"), getattr(controller.parent, "id", "?"))
                if controller.is_alive and (controller.parent is None or controller.parent.is_alive):
                    controller.stop()
                elif controller.is_alive:
                    logger.warning("TrafficSpawner: controlador (id=%s) con walker ya muerto "
                                    "(atropello a media escena); se omite stop() antes de destruir",
                                    getattr(controller, "id", "?"))
            except Exception:
                logger.warning("TrafficSpawner: fallo en stop() de un controlador de peatón (id=%s)",
                                getattr(controller, "id", "?"))

        # --- Fase 2: tick de seguridad ---
        if self._world is not None:
            try:
                self._world.tick()
            except Exception:
                pass

        # --- Fase 3: destruir ---
        for actor in self._controllers + self._walkers + self._vehicles:
            try:
                if actor.is_alive:
                    actor.destroy()
            except Exception:
                logger.warning("TrafficSpawner: fallo destruyendo actor id=%s tipo=%s (probablemente "
                                "el servidor ya lo habia eliminado)",
                                getattr(actor, "id", "?"), getattr(actor, "type_id", "?"))
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
