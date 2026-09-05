"""SensorSuite: monta y lee los sensores de CARLA para el vehículo ego.

Cámaras y LiDAR encolan sus lecturas crudas en una `queue.Queue` por sensor (el callback
solo hace `put`); `get_frame()` empareja por frame exacto (`data.frame`) como Roach/carla_garage,
para no mezclar en una misma observación datos de dos frames simulados distintos.

Convención de cámaras: nombres -> transform (x, y, z, yaw) respecto al ego.
"""

import queue
import time

import numpy as np

# Tipos de marca vial que sí penaliza lane_invasion. Nombres de carla.LaneMarkingType.
SOLID_LANE_MARKINGS = {"Solid", "SolidSolid"}


# Clasificación del actor impactado para el Infraction Penalty del Leaderboard: el
# coeficiente depende de contra qué se choca (peatón 0.50, vehículo 0.60, estático 0.65).
COLLISION_PREFIX = {"walker": "pedestrian", "vehicle": "vehicle"}


def _collision_type(event):
    """Traduce `event.other_actor.type_id` a "pedestrian"/"vehicle"/"static".

    Devuelve None si el evento no trae un actor utilizable (el DS lo tratará como estático,
    el coeficiente más suave de los tres; ver adcarla/carla_env/metrics.py).
    """
    type_id = getattr(getattr(event, "other_actor", None), "type_id", None)
    if not type_id:
        return None
    return COLLISION_PREFIX.get(str(type_id).split(".")[0], "static")


def _is_solid_lane_mark(lane_marking) -> bool:
    """Compara el tipo de marca por su nombre, no por el enum de carla (duck typing): así se
    puede fakear en tests sin importar carla, pasando cualquier objeto con `.type` o un string."""
    tipo = getattr(lane_marking, "type", lane_marking)
    nombre = getattr(tipo, "name", tipo)
    return str(nombre) in SOLID_LANE_MARKINGS


# Colocación aproximada de cámaras (metros, grados). Ajustable.
CAMERA_TRANSFORMS = {
    "front":        dict(x=1.5, y=0.0,  z=1.6, yaw=0.0),
    "front_left":   dict(x=1.2, y=-0.6, z=1.6, yaw=-55.0),
    "left":         dict(x=0, y=-0.6, z=1.6, yaw=-90.0),
    "front_right":  dict(x=1.2, y=0.6,  z=1.6, yaw=55.0),
    "right":        dict(x=0, y=0.6,  z=1.6, yaw=90.0),
    "rear":         dict(x=-1.5, y=0.0, z=1.6, yaw=180.0),
}


class SensorSuite:
    """Monta cámaras/LiDAR/colisión/invasión de línea en el ego y expone sus lecturas por frame."""

    def __init__(self, world, ego, config: dict):
        """Crea y adjunta al ego los sensores indicados en config["sensors"].

        Args:
            world: carla.World donde se spawnean los sensores.
            ego: actor vehículo al que se adjuntan los sensores.
            config: diccionario de configuración con la clave "sensors" (cameras,
                lidar, image_size). Sin cámaras configuradas no se monta ninguna: es el caso
                del profesor, que solo consume la máscara BEV privilegiada.

        Raises:
            RuntimeError: si hay cámaras configuradas pero falta "sensors.image_size".
        """
        import carla

        # Inicialización
        self.world = world
        self.ego = ego
        self.config = config
        self.sensors_config = config.get("sensors", {})
        self.camera_names = list(self.sensors_config.get("cameras", []))
        self._actors = []
        self._queues = {}   # nombre de sensor (camara o lidar) -> Queue de lecturas crudas
        # Flags de colisión e invasión de línea, más el tipo del actor impactado.
        self.events = {"collision": False, "lane_invasion": False, "collision_type": None}

        # Cadena _setup_cameras -> _setup_lidar -> _setup_events: si un paso posterior falla
        # tras haber spawneado ya sensores, hay que destruirlos antes de propagar la excepción
        # (si no, __init__ no termina, self.sensors se queda en None en env.py y esos sensores
        # ya spawneados quedan huérfanos, emitiendo el resto del proceso.
        try:
            bp = world.get_blueprint_library()
            if self.camera_names:
                if "image_size" not in self.sensors_config:
                    raise RuntimeError(
                        "SensorSuite: falta 'sensors.image_size' en el config. La resolución de captura "
                        "la fija el YAML ([160, 288] en los alumnos), no el código: capturar a otra "
                        "resolución cambia el numero de tokens de la spatial cross-attention.")
                camera_height, camera_width = self.sensors_config["image_size"]
                self._img_height_width = (int(camera_height), int(camera_width))
                self._setup_cameras(carla, bp)

            if self.sensors_config.get("lidar", False):
                self._setup_lidar(carla, bp)

            self._setup_events(carla, bp)
        except Exception:
            try:
                self.destroy()
            except Exception:
                pass   # la excepcion de la limpieza no debe enmascarar la original
            raise

    # ---- SETUP ----
    def _cam_transform(self, carla, camera_name):
        """Convierte una entrada del tipo de cámara en una posición relativa al ego."""
        transform = CAMERA_TRANSFORMS[camera_name]
        return carla.Transform(carla.Location(x=transform["x"], y=transform["y"], z=transform["z"]),
                               carla.Rotation(yaw=transform["yaw"]))

    def _setup_cameras(self, carla, bp):
        """Crea las cámaras RGB configuradas en config["sensors"]["cameras"]."""
        # Configuración (blueprint)
        camera_height, camera_width = self._img_height_width
        cam_bp = bp.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(camera_width))
        cam_bp.set_attribute("image_size_y", str(camera_height))
        cam_bp.set_attribute("fov", "90")

        # Creación
        for camera_name in self.camera_names:
            q = queue.Queue()
            self._queues[camera_name] = q
            sensor = self.world.spawn_actor(cam_bp, self._cam_transform(carla, camera_name), attach_to=self.ego)
            sensor.listen(q.put)  # el callback solo encola: decodificar aqui reintroduciria la carrera
            self._actors.append(sensor)

    def _setup_lidar(self, carla, bp):
        """Configuración y creación del LiDAR."""
        # Configuración (blueprint)
        lidar_bp = bp.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("range", "50")
        lidar_bp.set_attribute("rotation_frequency", "10")
        lidar_bp.set_attribute("channels", "32")    # Numero de lasers
        lidar_bp.set_attribute("points_per_second", "300000")

        # Creación
        # z = 1.8 m: es la referencia de Z_MIN/Z_MAX en encoders/fusion/lidar_branch.py, que
        # normaliza la altura del raster BEV respecto al sensor. Cambiar una cosa sin la otra
        # desplaza todo el canal de altura.
        transform = carla.Transform(carla.Location(x=0.0, z=1.8))
        q = queue.Queue()
        self._queues["lidar"] = q
        sensor = self.world.spawn_actor(lidar_bp, transform, attach_to=self.ego)
        sensor.listen(q.put)
        self._actors.append(sensor)

    def _setup_events(self, carla, bp):
        """Crea los sensores de colisión e invasión de línea."""
        # Sensor de colisión (para la función reward)
        collision_sensor = self.world.spawn_actor(bp.find("sensor.other.collision"),
                                     carla.Transform(), attach_to=self.ego)
        collision_sensor.listen(self._on_collision)

        # Sensor de invasión de línia (para la función reward)
        lane_sensor = self.world.spawn_actor(bp.find("sensor.other.lane_invasion"),
                                      carla.Transform(), attach_to=self.ego)
        lane_sensor.listen(self._on_lane_invasion)

        self._actors += [collision_sensor, lane_sensor]

    def _on_collision(self, event):
        """Marca la colisión y clasifica el actor impactado. Si en el mismo tick hay varios
        choques se conserva el primero: el episodio termina en la colisión, así que el segundo
        no llega a puntuar."""
        if not self.events["collision"]:
            self.events["collision_type"] = _collision_type(event)
        self.events["collision"] = True

    def _on_lane_invasion(self, event):
        """Solo cuenta como infracción cruzar una marca continua: una discontinua es
        un cambio de carril legal, no una infracción."""
        marks = getattr(event, "crossed_lane_markings", [])
        if any(_is_solid_lane_mark(mark) for mark in marks):
            self.events["lane_invasion"] = True

    # ---- DECODIFICACIÓN ----
    @staticmethod
    def _decode_camera(image):
        """Decodifica el frame BGRA de carla.Image a un array RGB (copia, sin canal alfa)."""
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3]  # BGRA -> BGR, descarta Alpha
        return arr[:, :, ::-1].copy()  # BGR -> RGB, alterna el orden porque las otras librerias tratan imágenes como RGB

    @staticmethod
    def _decode_lidar(data):
        """Decodifica la nube de puntos (x, y, z, intensity) de carla.LidarMeasurement."""
        pts = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
        return pts.copy()

    # ---- API ----
    def _wait_for_frame(self, name: str, frame: int, timeout: float):
        """Descarta lecturas de la cola `name` hasta encontrar `data.frame == frame`.

        Descarta tanto frames atrasados como cualquier otro que no coincida (por si llegasen
        desordenados), sin bloquear más allá de `timeout` en total.

        Raises:
            TimeoutError: si el frame pedido no llega dentro de `timeout` segundos.
        """
        q = self._queues[name]
        deadline = time.monotonic() + timeout
        while True:
            restante = deadline - time.monotonic()
            if restante <= 0:
                raise TimeoutError(
                    f"SensorSuite: el sensor '{name}' no entrego el frame {frame} en {timeout}s")
            try:
                data = q.get(timeout=restante)
            except queue.Empty:
                raise TimeoutError(
                    f"SensorSuite: el sensor '{name}' no entrego el frame {frame} en {timeout}s")
            if data.frame == frame:
                return data
            # frame distinto del pedido (atrasado o desordenado): se descarta y se sigue buscando

    def get_frame(self, frame: int, timeout: float = 2.0) -> dict:
        """Observación con TODOS los sensores emparejados exactamente al mismo `frame` simulado.

        Args:
            frame: frame de simulación a buscar (world.get_snapshot().frame tras el tick).
            timeout: segundos máximos de espera por sensor.

        Returns:
            {"cameras": {nombre: array RGB}, "lidar": array [N, 4] opcional}, ya decodificado.
        """
        cams = {name: self._decode_camera(self._wait_for_frame(name, frame, timeout))
                for name in self.camera_names}
        obs = {"cameras": cams}
        if "lidar" in self._queues:
            obs["lidar"] = self._decode_lidar(self._wait_for_frame("lidar", frame, timeout))
        return obs

    def clear_queues(self):
        """Vacía todas las colas sin bloquear (usado tras el calentamiento de `reset()`)."""
        for q in self._queues.values():
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def pop_events(self) -> dict:
        """Devuelve los eventos (colisión, invasión de línea) acumulados y resetea los flags."""
        ev = dict(self.events)
        self.events["collision"] = False
        self.events["lane_invasion"] = False
        self.events["collision_type"] = None
        return ev

    def destroy(self):
        """Detiene y destruye todos los actores sensor creados."""
        for a in self._actors:
            try:
                a.stop()
            except Exception:
                pass
            try:
                a.destroy()
            except Exception:
                pass
        self._actors = []
