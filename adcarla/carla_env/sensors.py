"""SensorSuite: monta y lee los sensores de CARLA para el vehículo ego.

Cada sensor registra un callback que guarda la ÚLTIMA lectura, `get_obs()` las recoge.
Depende de CARLA (import perezoso).

Convención de cámaras: nombres -> transform (x, y, z, yaw) respecto al ego.
"""

import numpy as np

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
    """Monta cámaras/LiDAR/colisión/invasión de línea en el ego y expone sus últimas lecturas."""

    def __init__(self, world, ego, config: dict):
        """Crea y adjunta al ego los sensores indicados en config["sensors"].

        Args:
            world: carla.World donde se spawnean los sensores.
            ego: actor vehículo al que se adjuntan los sensores.
            config: diccionario de configuración con la clave "sensors" (cameras,
                lidar, image_size).
        """
        import carla

        # Inicialización
        self.world = world
        self.ego = ego
        self.config = config
        self.sensors_config = config.get("sensors", {})
        self._actors = []
        self._latest = {}   # diccionario última observación {nombre -> np.array}
        self.events = {"collision": False, "lane_invasion": False}  # Flags colisión, invasión línea

        # Configuración cámaras
        bp = world.get_blueprint_library()
        camera_height, camera_width = self.sensors_config.get("image_size", [256, 448])
        self._img_height_width = (int(camera_height), int(camera_width))
        self._setup_cameras(carla, bp)

        # Configuración LiDAR
        if self.sensors_config.get("lidar", False):
            self._setup_lidar(carla, bp)
        
        self._setup_events(carla, bp)

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
        for camera_name in self.sensors_config.get("cameras", ["front"]):
            sensor = self.world.spawn_actor(cam_bp, self._cam_transform(carla, camera_name), attach_to=self.ego)
            sensor.listen(lambda data, name=camera_name: self._on_camera(data, name))
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
        transform = carla.Transform(carla.Location(x=0.0, z=1.8))
        sensor = self.world.spawn_actor(lidar_bp, transform, attach_to=self.ego)
        sensor.listen(lambda data: self._on_lidar(data))
        self._actors.append(sensor)

    def _setup_events(self, carla, bp):
        """Crea los sensores de colisión e invasión de línea."""
        # Sensor de colisión (para la función reward)
        collision_sensor = self.world.spawn_actor(bp.find("sensor.other.collision"),
                                     carla.Transform(), attach_to=self.ego)
        collision_sensor.listen(lambda e: self.events.__setitem__("collision", True))

        # Sensor de invasión de línia (para la función reward)
        lane_sensor = self.world.spawn_actor(bp.find("sensor.other.lane_invasion"),
                                      carla.Transform(), attach_to=self.ego)
        lane_sensor.listen(lambda e: self.events.__setitem__("lane_invasion", True))

        self._actors += [collision_sensor, lane_sensor]

    # ---- CALLBACKS ----
    def _on_camera(self, image, camera_name):
        """Callback de carla.Sensor: decodifica el frame BGRA y lo guarda en RGB como última lectura."""
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3] # BGRA -> BGR, descarta Alpha (transparencia)
        self._latest[camera_name] = arr[:, :, ::-1].copy() # BGR -> RGB, alterna el orden porque las otras librerias tratan imágenes como RGB

    def _on_lidar(self, data):
        """Callback de carla.Sensor: decodifica la nube de puntos (x, y, z, intensity)."""
        pts = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)  # x,y,z,intensity (coordenadas y intensidad que sirve como medida de que tan lejos está el punto)
        self._latest["lidar"] = pts.copy()

    # ---- API ----
    def get_obs(self) -> dict:
        """Última observación disponible: {"cameras": {nombre_camera: array}, "lidar": array opcional}."""
        cams = {name: self._latest.get(name) for name in self.sensors_config.get("cameras", ["front"])}
        obs = {"cameras": cams}
        if self.sensors_config.get("lidar", False):
            obs["lidar"] = self._latest.get("lidar")
        return obs

    def pop_events(self) -> dict:
        """Devuelve los eventos (colisión, invasión de línea) acumulados y resetea los flags."""
        ev = dict(self.events)
        self.events["collision"] = False
        self.events["lane_invasion"] = False
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
