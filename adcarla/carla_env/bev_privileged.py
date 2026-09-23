"""PrivilegedBEVGenerator: máscara BEV semántica privilegiada del profesor.

Salida: array 3D float32 [C, H, W] con valores en {0,1}, ego en el centro, "delante" = hacia arriba).
C = Channel, H = Height, W = Width
Estilo Think2Drive/Roach: cada canal es una capa semántica/un mapa binario con el que se marca si en esa
coordenada existe o no el elemento del canal.
"""

import math
import numpy as np

# Canales de la máscara (el orden define el índice). len(CHANNELS) = bev.channels.
CHANNELS = ["road", "route", "ego", "vehicle", "pedestrian", "light_green", "light_red"]
CHANNEL_INDEX = {name: i for i, name in enumerate(CHANNELS)}


class PrivilegedBEVGenerator:
    """Genera la máscara BEV semántica privilegiada del profesor."""

    def __init__(self, cfg: dict):
        """Fija resolución y cobertura de la rejilla BEV a partir de cfg["bev"].

        Args:
            cfg: configuración completa, se usa la clave "bev" (size, range_meters) del diccionario.
        """
        b = cfg.get("bev", {})
        self.size = int(b.get("size", 128))            # Height = Width
        self.range_meters = float(b.get("range_meters", 50.0))   # semilado del área BEV cuadrada
        self.channels = CHANNELS
        self.n_channels = len(CHANNELS)
        self.mpp = (2.0 * self.range_meters) / self.size    # metros por píxel

    ######################################################################
    #############################  GEOMETRÍA  ############################
    ######################################################################
    def world_to_bev(self, pts_xy: np.ndarray, ego) -> np.ndarray:
        """Proyecta puntos del mundo a píxeles de la rejilla BEV, centrada en el ego y orientada según su yaw.

        Traslada los puntos al origen del ego, los rota a su marco de referencia local (delante/derecha) y
        convierte esos metros a píxeles.

        Args:
            pts_xy: (N,2) puntos en coordenadas mundo (x,y), en metros.
            ego: dict con la pose del ego: x, y (metros) y yaw (grados).

        Returns:
            (N,2) puntos en píxeles (col, row) dentro de la rejilla BEV.
        """
        pts = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 2)   # conversión a array numpy con dos columnas (x,y)

        # Cálculo de la posición de los puntos al marco de referencia del ego
        dx = pts[:, 0] - ego["x"]
        dy = pts[:, 1] - ego["y"]
        yaw_rad = math.radians(ego["yaw"])
        cos_yaw_rad, sin_yaw_rad = math.cos(yaw_rad), math.sin(yaw_rad)

        fx = dx * cos_yaw_rad + dy * sin_yaw_rad          # eje x -> longitudinal
        fy = -dx * sin_yaw_rad + dy * cos_yaw_rad         # eje y -> lateral
        col = self.size / 2.0 + fy / self.mpp
        row = self.size / 2.0 - fx / self.mpp   # delante = hacia arriba (row menor)

        return np.stack([col, row], axis=1)

    @staticmethod
    def box_corners(x, y, yaw_deg, ex, ey) -> np.ndarray:
        """Devuelve las 4 esquinas (mundo) de una caja centrada en (x,y), con inclinación yaw (grados) y
        la mitad de la longitud de sus lados(ex,ey)."""
        yaw_rad = math.radians(yaw_deg)
        cos_yaw_rad, sin_yaw_rad = math.cos(yaw_rad), math.sin(yaw_rad)

        corners = np.array([[ex, ey], [ex, -ey], [-ex, -ey], [-ex, ey]], dtype=np.float32)    # 4 esquinas de un rectángulo centrado en el origen (0,0), sin rotar
        rotation_matrix = np.array([[cos_yaw_rad, -sin_yaw_rad], [sin_yaw_rad, cos_yaw_rad]], dtype=np.float32)   # matriz de rotación 2D

        corners_rotated = corners @ rotation_matrix.T     # rotación de las cuatro esquinas centradas en (0,0)

        return corners_rotated + np.array([x, y], dtype=np.float32)     # translación de las cuatro esquinas alrededor del punto (x,y)

    def _fill_convex(self, mask: np.ndarray, poly_px: np.ndarray, value: float = 1.0):
        """Rasteriza (rellena) un polígono convexo sobre mask, dado en píxeles (col,row).
        Es decir, pinta polígonos con coordenadas del mundo en un canal BEV que es una grid binaria.

        Args:
            mask: array (H, W), se modifica in-place. Mapa binario que define un canal BEV.
            poly_px: (K, 2) vértices consecutivos en píxeles (col, row). Esquinas de la figura
                a dibujar en coordenadas de la grid binaria.
            value: valor de relleno para los píxeles interiores.
        """
        poly = np.asarray(poly_px, dtype=np.float32)
        H, W = mask.shape

        # Límites del polígono recortado a mask. Se obtiene las coordenadas x e y mínimas y máximas y
        # se verifica que no sean más grandes que el ancho y alto de la grid BEV. Si es el caso, se
        # limitará el espacio en el que podemos escribir en la grid BEV.
        x0 = max(int(np.floor(poly[:, 0].min())), 0)
        y0 = max(int(np.floor(poly[:, 1].min())), 0)

        x1 = min(int(np.ceil(poly[:, 0].max())), W - 1)        
        y1 = min(int(np.ceil(poly[:, 1].max())), H - 1)

        if x1 < x0 or y1 < y0:
            return  # el polígono cae entero fuera de mask

        # Píxeles candidatos dentro de los límites, aplanados a listas 1D (col, row)
        # np.arange crea una lista de cada valor entero entre los dos valores
        # np.meshgrid combina esos dos vectores 1D en dos rejillas 2D
        # Ejemplo: si x0,x1 = 2,4 (columnas 2,3,4) e y0,y1 = 5,6 (filas 5,6):
        # gx = [[2,3,4], [2,3,4]]       gy = [[5,5,5], [6,6,6]]

        gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))

        # ravel() aplana las arrays 2D a una 1D, concatenandolas
        # Ejemplo:  gx.ravel() = [2, 3, 4, 2, 3, 4]
        #           gy.ravel() = [5, 5, 5, 6, 6, 6]
        px = gx.ravel().astype(np.float32)
        py = gy.ravel().astype(np.float32)

        # Producto cruzado punto candidato-arista (una fila por arista, vectorizado sobre los candidatos)
        # para determinar de que lado de cada arista está el punto
        # poly[(i + 1) % K] - poly[i]) || Vector de la arista (del vertice i al vertice i + 1)
        # (py - poly[i]) || Vector del vertice i a un punto de la grid que hemos hecho en el paso anterior
        K = len(poly)
        cross = np.stack([
            (poly[(i + 1) % K, 0] - poly[i, 0]) * (py - poly[i, 1]) -
            (poly[(i + 1) % K, 1] - poly[i, 1]) * (px - poly[i, 0])
            for i in range(K)
        ], axis=0)

        # Dentro del polígono = mismo signo (mismo lado) en relación a todas las aristas a la vez
        points_inside_poly = np.all(cross >= -1e-6, axis=0) | np.all(cross <= 1e-6, axis=0) # array booleano de si los puntos de la grid están dentro
        points_index = np.where(points_inside_poly)[0]  # index del array de los puntos que estan dentro del poligono

        # Relleno en la máscara (mapa binario) de los puntos dentro del polígono
        mask[py[points_index].astype(int), px[points_index].astype(int)] = value

    def _draw_box(self, channel_mask, x, y, yaw, ex, ey, ego):
        """Dibuja una caja orientada (mundo) sobre channel_mask: calcula esquinas, proyecta a BEV y rellena."""
        corners = self.box_corners(x, y, yaw, ex, ey)
        self._fill_convex(channel_mask, self.world_to_bev(corners, ego))

    def _draw_polyline(self, channel_mask, pts_world, ego, thick=1.2):
        """Dibuja una línea gruesa (la ruta) como quads entre puntos consecutivos."""
        pts = np.asarray(pts_world, dtype=np.float32).reshape(-1, 2) # normaliza a un array (N,2), N puntos de ruta con sus 2 coordenadas (x,y)

        # Iteramos sobre dos arrays de puntos consecutivos del polígono
        for a, b in zip(pts[:-1], pts[1:]):
            vector = b - a
            norm = np.linalg.norm(vector) # norma euclídea

            # Si la distancia es ínfima, saltamos la iteración
            if norm < 1e-3:
                continue

            perpendicular = np.array([-vector[1], vector[0]]) / norm * thick # vector perpendicular del ancho deseado
            quad = np.array([a + perpendicular, b + perpendicular, b - perpendicular, a - perpendicular], dtype=np.float32) # cuatro esquinas del rectángulo a pitnar

            self._fill_convex(channel_mask, self.world_to_bev(quad, ego)) # pintamos el rectángulo

    ######################################################################
    ##############################  RENDER  ##############################
    ######################################################################
    def render(self, ego, actors=None, route_pts=None, lights=None, road_quads=None) -> np.ndarray:
        """Construye representación BEV [C,H,W] a partir de datos YA extraídos (todo en coordenadas mundo).
        
        Args:
            ego: posición vehículo ego | dict(x,y,yaw,ex,ey).
            actors: actores del mundo carla, con su posicion y tipo | [dict(x,y,yaw,ex,ey,kind)].
            route_pts: puntos de la ruta | [(x,y)].
            lights: semáforos, con su posicion y estado | [dict(x,y,state)].
            road_quads: rectángulos que forman la carretera | [(4,2) esquinas mundo].
        """
        mask_bev = np.zeros((self.n_channels, self.size, self.size), dtype=np.float32)  # array 3D lleno de zeros

        # Pinta carreteras
        for road_quad in (road_quads or []):
            self._fill_convex(mask_bev[CHANNEL_INDEX["road"]], self.world_to_bev(road_quad, ego))

        # Pinta la ruta
        if route_pts is not None and len(route_pts) >= 2:
            self._draw_polyline(mask_bev[CHANNEL_INDEX["route"]], route_pts, ego)

        # Pinta el vehículo ego
        self._draw_box(mask_bev[CHANNEL_INDEX["ego"]], ego["x"], ego["y"], ego["yaw"], ego["ex"], ego["ey"], ego)

        # Pinta los actores (peatones y otros vehículos)
        for actor in (actors or []):
            channel = CHANNEL_INDEX["pedestrian"] if actor.get("kind") == "pedestrian" else CHANNEL_INDEX["vehicle"]
            self._draw_box(mask_bev[channel], actor["x"], actor["y"], actor["yaw"], actor["ex"], actor["ey"], ego)

        # Pinta los semáforos
        for light in (lights or []):
            channel = CHANNEL_INDEX["light_green"] if light.get("state") == "green" else CHANNEL_INDEX["light_red"]
            self._draw_box(mask_bev[channel], light["x"], light["y"], 0.0, 1.0, 1.0, ego)

        return mask_bev

    ######################################################################
    ####################### EXTRACCIÓN DESDE CARLA #######################
    ######################################################################
    def generate(self, world, ego_actor, route_pts=None) -> np.ndarray:
        """Extrae actores/mapa/semáforos de CARLA alrededor del ego y llama a render()."""
        import carla

        ego_transform = ego_actor.get_transform()
        ego_location = ego_transform.location
        ego_bounding_box = ego_actor.bounding_box.extent
        ego = {"x": ego_location.x, "y": ego_location.y, "yaw": ego_transform.rotation.yaw, "ex": ego_bounding_box.x, "ey": ego_bounding_box.y}

        def near(location):
            return abs(location.x - ego_location.x) <= self.range_meters and abs(location.y - ego_location.y) <= self.range_meters

        # Guada los actores (coches y peatones) en un array
        actors = []
        for vehicle in world.get_actors().filter("vehicle.*"):
            if vehicle.id == ego_actor.id:
                continue

            vehicle_transform = vehicle.get_transform()
            if near(vehicle_transform.location):
                vehicle_bounding_box = vehicle.bounding_box.extent
                actors.append({"x": vehicle_transform.location.x, "y": vehicle_transform.location.y, "yaw": vehicle_transform.rotation.yaw,
                               "ex": vehicle_bounding_box.x, "ey": vehicle_bounding_box.y, "kind": "vehicle"})

        for pedestrian in world.get_actors().filter("walker.pedestrian.*"):
            pedestrian_transform = pedestrian.get_transform()
            if near(pedestrian_transform.location):
                pedestrian_bounding_box = pedestrian.bounding_box.extent
                actors.append({"x": pedestrian_transform.location.x, "y": pedestrian_transform.location.y, "yaw": pedestrian_transform.rotation.yaw,
                               "ex": pedestrian_bounding_box.x, "ey": pedestrian_bounding_box.y, "kind": "pedestrian"})

        # Guarda semáforos en un array
        lights = []
        for traffic_light in world.get_actors().filter("traffic.traffic_light*"):
            light_transform = traffic_light.get_transform()
            if near(light_transform.location):
                traffic_light_state = str(traffic_light.get_state())
                lights.append({"x": light_transform.location.x, "y": light_transform.location.y,
                               "state": "green" if "Green" in traffic_light_state else "red"})

        road_quads = self._road_quads(world.get_map(), ego_location)
        route_world = [(p.x, p.y) for p in (route_pts or [])]

        return self.render(ego, actors=actors, route_pts=route_world,
                           lights=lights, road_quads=road_quads)

    def _road_quads(self, carla_map, ego_location, step=2.0, max_waypoints=400):
        """Quads de la superficie de calzada alrededor del ego.

        La calzada es estática: no cambia dentro de un episodio ni entre episodios del mismo
        mapa. Se precalcula una vez por mapa y se indexa por celdas (estilo
        Roach, que rasteriza el mapa entero al cargarlo), el tick solo consulta las celdas vecinas.
        """
        cache = _road_quad_cache(carla_map, step)

        return cache.near(ego_location.x, ego_location.y, self.range_meters, max_waypoints)


_ROAD_QUAD_CACHES = {}
ROAD_CELL_METERS = 25.0   # lado de la celda del índice espacial de la caché de calzada


def _road_quad_cache(carla_map, step: float):
    """Devuelve (creándola la primera vez) la caché de calzada del mapa, compartida por proceso."""
    clave = (getattr(carla_map, "name", None) or id(carla_map), step)
    if clave not in _ROAD_QUAD_CACHES:
        _ROAD_QUAD_CACHES[clave] = RoadQuadCache(carla_map, step)

    return _ROAD_QUAD_CACHES[clave]


class RoadQuadCache:
    """Quads de calzada de un mapa entero, calculados una vez e indexados por celdas."""

    def __init__(self, carla_map, step: float = 2.0, cell_meters: float = ROAD_CELL_METERS):
        """Muestrea todos los carriles conducibles del mapa y reparte sus quads en celdas.

        Args:
            carla_map: objeto devuelto por `world.get_map()`.
            step: separación entre waypoints, y también la longitud de cada quad.
            cell_meters: lado de la celda del índice espacial.
        """
        import carla

        self.cell_meters = float(cell_meters)
        self.tiles = {}
        half_length = step / 2.0

        for waypoint in carla_map.generate_waypoints(step):
            if waypoint.lane_type != carla.LaneType.Driving:
                continue

            transform = waypoint.transform
            x, y = transform.location.x, transform.location.y
            # Quad centrado en el waypoint (largada `step`, anchura del carril) en vez de
            # waypoint->next(): misma cobertura y sin una llamada más a CARLA por waypoint.
            quad = PrivilegedBEVGenerator.box_corners(x, y, transform.rotation.yaw,
                                                      half_length, waypoint.lane_width / 2.0)
            self.tiles.setdefault(self._key(x, y), []).append((x, y, quad))

    def _key(self, x: float, y: float):
        return (int(math.floor(x / self.cell_meters)), int(math.floor(y / self.cell_meters)))

    def near(self, x: float, y: float, range_meters: float, max_quads: int) -> list:
        """Quads cuyo centro cae en el cuadrado de semilado `range_meters` alrededor de (x, y).

        Las celdas se recorren de más cerca a más lejos, así que si se alcanza `max_quads` lo
        que se descarta es siempre lo más lejano.
        """
        radio = int(math.ceil(range_meters / self.cell_meters))
        cx, cy = self._key(x, y)
        celdas = [(i, j) for i in range(cx - radio, cx + radio + 1)
                  for j in range(cy - radio, cy + radio + 1)]
        celdas.sort(key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2)

        quads = []
        for celda in celdas:
            for qx, qy, quad in self.tiles.get(celda, ()):
                if abs(qx - x) <= range_meters and abs(qy - y) <= range_meters:
                    quads.append(quad)
                    if len(quads) >= max_quads:
                        return quads

        return quads
