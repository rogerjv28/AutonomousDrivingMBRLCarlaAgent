"""RouteManager: rutas fijas y reproducibles para la comparativa entre ramas.

Sustituye al spawn aleatorio de `CarlaEnv.reset()`: sin rutas fijas, una diferencia de reward
entre ramas podría venir del mapa, no del sensor. Si `scenarios.routes` está definida en el YAML se usa
tal cual y se ignoran `num_*_routes`.

Diseño: la lógica de qué ruta toca en cada episodio es pura para poder testearla sin simulador,
solo `spawn_transform()` y `build_route()` llaman la API de CARLA, y reciben el mapa como argumento.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from typing import Optional

from .route_tracker import RouteWaypoint, command_from_road_option


@dataclass(frozen=True)
class RouteSpec:
    """Definición de una ruta reproducible.

    Attributes:
        route_id: identificador legible, aparece en los logs y en las métricas.
        spawn_index: índice dentro de `carla_map.get_spawn_points()`. Es lo que hace
            la ruta reproducible: el mismo índice da siempre el mismo punto de partida.
        destination_index: índice del spawn point de destino. El GlobalRoutePlanner traza
            el camino entre ambos.
        length_waypoints: tope de waypoints (de `sampling_resolution` metros) de la ruta,
            si el destino queda más lejos, la ruta se trunca ahí.
        max_steps: corte por tiempo del episodio, en pasos de simulación.
        split: "train" o "eval".
        weather: nombre de un preset de carla.WeatherParameters, o None para no tocarlo.
    """
    route_id: str
    spawn_index: int
    destination_index: int = 0
    length_waypoints: int = 200
    max_steps: int = 600
    split: str = "train"
    weather: Optional[str] = None

    def as_dict(self) -> dict:
        """Representación serializable (para logs y checkpoints)."""
        return asdict(self)


DEFAULTS = {
    "num_train_routes": 8,
    "num_eval_routes": 4,
    "length_waypoints": 200,
    "max_steps": 600,
    "seed": 0,
    "sampling_resolution": 2.0,
    "routes": None,
}


class RouteManager:
    """Reparte rutas fijas entre episodios de forma determinista.

    El mismo `config` y el mismo número de spawn points producen siempre el mismo
    conjunto de rutas y el mismo orden, así que las dos ramas del TFM recorren
    exactamente los mismos trayectos en el mismo orden.
    """

    def __init__(self, config: dict, num_spawn_points: int):
        """Construye (o carga) el conjunto de rutas.

        Args:
            config: configuración completa, se lee el bloque "scenarios".
            num_spawn_points: `len(carla_map.get_spawn_points())`. Se pasa como argumento
                para que la clase sea testeable sin CARLA.

        Raises:
            ValueError: si no hay spawn points suficientes para las rutas pedidas.
        """
        scenarios_config = {**DEFAULTS, **config.get("scenarios", {})}
        self.config = scenarios_config
        self.num_spawn_points = int(num_spawn_points)

        explicit_routes = scenarios_config.get("routes")
        if explicit_routes:
            self._routes = [self._spec_from_dict(route, scenarios_config) for route in explicit_routes]
        else:
            self._routes = self._generate(scenarios_config)

        self._validate()

    # ---- CONSTRUCCIÓN ----
    @staticmethod
    def _spec_from_dict(raw: dict, defaults: dict) -> RouteSpec:
        """Convierte una entrada de `scenarios.routes` del YAML en un RouteSpec.

        Raises:
            ValueError: si falta `spawn_index` o `destination_index` (sin destino no hay nada
                que trazar, y el fallo aparecería a mitad del entrenamiento).
        """
        for point_index in ("spawn_index", "destination_index"):
            if point_index not in raw:
                raise ValueError(f"A la ruta {raw.get('id', raw)} de 'scenarios.routes' le falta '{point_index}'")

        return RouteSpec(
            route_id=str(raw.get("id", f"spawn{raw['spawn_index']}")),
            spawn_index=int(raw["spawn_index"]),
            destination_index=int(raw["destination_index"]),
            length_waypoints=int(raw.get("length_waypoints", defaults["length_waypoints"])),
            max_steps=int(raw.get("max_steps", defaults["max_steps"])),
            split=str(raw.get("split", "train")),
            weather=raw.get("weather"),
        )

    def _generate(self, scenarios_config: dict) -> list:
        """Sortea spawn points de forma determinista y los reparte en train/eval disjuntos."""
        num_train = int(scenarios_config["num_train_routes"])
        num_eval = int(scenarios_config["num_eval_routes"])
        total = num_train + num_eval

        # Cada ruta consume dos spawn points (origen y destino) y ninguno se repite: así los
        # tramos de eval no se solapan con los de train.
        if 2 * total > self.num_spawn_points:
            raise ValueError(
                f"Se piden {total} rutas ({num_train} train + {num_eval} eval), que necesitan "
                f"{2 * total} spawn points (origen + destino), pero el mapa solo tiene "
                f"{self.num_spawn_points}."
            )

        # random.Random(seed) aísla el sorteo del estado global: no depende de lo que haya
        # hecho antes numpy/torch, así que el reparto es idéntico en las dos ramas.
        rng = random.Random(int(scenarios_config["seed"]))
        chosen = rng.sample(range(self.num_spawn_points), 2 * total)

        routes = []
        for position in range(total):
            spawn_index, destination_index = chosen[2 * position], chosen[2 * position + 1]
            split = "train" if position < num_train else "eval"
            index_in_split = position if split == "train" else position - num_train

            routes.append(RouteSpec(
                route_id=f"{split}{index_in_split:02d}_spawn{spawn_index}",
                spawn_index=spawn_index,
                destination_index=destination_index,
                length_waypoints=int(scenarios_config["length_waypoints"]),
                max_steps=int(scenarios_config["max_steps"]),
                split=split,
            ))

        return routes

    def _validate(self):
        """Comprueba que los índices caen dentro del mapa y que no hay ids repetidos."""
        for spec in self._routes:
            for name, index in (("spawn_index", spec.spawn_index),
                                   ("destination_index", spec.destination_index)):
                if not 0 <= index < self.num_spawn_points:
                    raise ValueError(
                        f"Ruta '{spec.route_id}': {name} {index} fuera de rango "
                        f"[0, {self.num_spawn_points})."
                    )
            if spec.length_waypoints < 2:
                raise ValueError(f"Ruta '{spec.route_id}': length_waypoints debe ser >= 2 y es "
                                 f"{spec.length_waypoints}; una ruta de un punto no se puede recorrer.")
            if spec.spawn_index == spec.destination_index:
                raise ValueError(f"Ruta '{spec.route_id}': origen y destino son el mismo punto "
                                 f"({spec.spawn_index}); no hay ruta que trazar.")
        ids = [spec.route_id for spec in self._routes]
        if len(set(ids)) != len(ids):
            raise ValueError("Hay route_id repetidos en scenarios.routes")
        if not self._routes:
            raise ValueError("RouteManager se ha quedado sin rutas, revisa el bloque 'scenarios' del config")

    # ---- SELECCIÓN ----
    def routes(self, split: str = "train") -> list:
        """Devuelve la lista de RouteSpec de un split ("train" o "eval")."""
        return [spec for spec in self._routes if spec.split == split]

    def num_routes(self, split: str = "train") -> int:
        """Número de rutas disponibles en el split."""
        return len(self.routes(split))

    def select(self, episode: int, split: str = "train") -> RouteSpec:
        """Ruta que le toca al episodio `episode`, en round-robin determinista.

        Args:
            episode: contador de episodios del bucle de entrenamiento.
            split: "train" o "eval".

        Returns:
            El RouteSpec correspondiente.

        Raises:
            ValueError: si el split no tiene ninguna ruta.
        """
        available = self.routes(split)
        if not available:
            raise ValueError(f"No hay rutas definidas para el split '{split}'")
        
        return available[int(episode) % len(available)]

    # ---- INTERACCIÓN CON CARLA ----
    def spawn_transform(self, carla_map, spec: RouteSpec):
        """Devuelve el `carla.Transform` de partida de la ruta.

        Args:
            carla_map: objeto devuelto por `world.get_map()`.
            spec: ruta seleccionada.
        """
        return self._transform_at(carla_map, spec.spawn_index, spec)

    def destination_transform(self, carla_map, spec: RouteSpec):
        """Devuelve el `carla.Transform` de destino de la ruta."""
        return self._transform_at(carla_map, spec.destination_index, spec)

    @staticmethod
    def _transform_at(carla_map, index: int, spec: RouteSpec):
        """Spawn point `index` del mapa cargado.

        Raises:
            RuntimeError: si el mapa cargado tiene menos spawn points de los esperados
                (indica que el config apunta a otra ciudad).
        """
        spawn_points = carla_map.get_spawn_points()
        if index >= len(spawn_points):
            raise RuntimeError(
                f"El mapa cargado tiene {len(spawn_points)} spawn points, pero la ruta "
                f"'{spec.route_id}' pide el índice {index}. ¿Coincide 'carla.town' "
                f"con el mapa usado al generar las rutas?"
            )
        
        return spawn_points[index]

    def build_planner(self, carla_map):
        """Construye el `GlobalRoutePlanner` de CARLA con la resolución del config.

        Raises:
            ImportError: si `agents.navigation` no está en el PYTHONPATH (viene en
                `CARLA_ROOT/PythonAPI/carla`, junto al egg del cliente).
        """
        try:
            from agents.navigation.global_route_planner import GlobalRoutePlanner
        except ImportError as exc:
            raise ImportError(
                "No se encuentra 'agents.navigation.global_route_planner'. Añade "
                "CARLA_ROOT/PythonAPI/carla al PYTHONPATH (viene con el cliente de CARLA)."
            ) from exc

        return GlobalRoutePlanner(carla_map, float(self.config["sampling_resolution"]))

    def build_route(self, carla_map, spec: RouteSpec, planner=None) -> list:
        """Traza la ruta del origen al destino del spec y la devuelve sin objetos de CARLA.

        Args:
            carla_map: objeto devuelto por `world.get_map()`.
            spec: ruta seleccionada (origen, destino y `length_waypoints`).
            planner: `GlobalRoutePlanner` ya construido. Si es None se crea uno (permite
                reutilizarlo entre episodios y probar esto sin CARLA).

        Returns:
            Lista de `RouteWaypoint` (x, y, comando de navegación), truncada a
            `spec.length_waypoints`.

        Raises:
            RuntimeError: si el planner no encuentra camino entre los dos puntos.
        """
        planner = planner or self.build_planner(carla_map)
        start = self.spawn_transform(carla_map, spec).location
        end = self.destination_transform(carla_map, spec).location

        # La ruta se comprueba antes de truncar: si no, un length_waypoints pequeño daría
        # el error de "no ha encontrado ruta" sobre una ruta que el planner sí encontró.
        route = planner.trace_route(start, end)
        if len(route) < 2:
            raise RuntimeError(
                f"El GlobalRoutePlanner no ha encontrado ruta para '{spec.route_id}' "
                f"(spawn {spec.spawn_index} -> destino {spec.destination_index}): "
                f"{len(route)} waypoint(s)."
            )

        return [RouteWaypoint(waypoint.transform.location.x, waypoint.transform.location.y,
                              command_from_road_option(road_option))
                for waypoint, road_option in route[:int(spec.length_waypoints)]]

    @staticmethod
    def apply_weather(world, spec: RouteSpec, weather: str = None):
        """Aplica un preset de clima. No hace nada si ni `weather` ni `spec.weather` lo fijan.

        Args:
            world: mundo de CARLA.
            spec: ruta seleccionada (su `weather` es el clima por defecto de la ruta).
            weather: preset que pisa el de la ruta. Lo usa la evaluación fuera de distribución
                 que recorre las mismas rutas bajo varios climas.

        Raises:
            ValueError: si el nombre del preset no existe en carla.WeatherParameters.
        """
        name = weather or spec.weather
        if not name:
            return

        import carla
        preset = getattr(carla.WeatherParameters, name, None)
        if preset is None:
            raise ValueError(f"Preset de clima desconocido: '{name}'")
        world.set_weather(preset)

    def summary(self) -> dict:
        """Resumen serializable del conjunto de rutas (para guardar junto a los resultados)."""
        return {
            "num_spawn_points": self.num_spawn_points,
            "seed": self.config["seed"],
            "train": [spec.as_dict() for spec in self.routes("train")],
            "eval": [spec.as_dict() for spec in self.routes("eval")],
        }
