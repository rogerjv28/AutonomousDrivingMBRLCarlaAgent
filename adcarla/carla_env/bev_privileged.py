"""PrivilegedBEVGenerator: máscara BEV semántica privilegiada del profesor.

Salida: array 3D float32 [C, H, W] con valores en {0,1}, ego en el centro, "delante" = hacia arriba).
C = Channel, H = Height, W = Width
Estilo Think2Drive/Roach: cada canal es una capa semántica/un mapa binario con el que se marca si en esa
coordenada existe o no el elemento del canal.

Diseño para testeo: la geometría y el rasterizado son numpy puro (testeables sin CARLA);
la extracción desde CARLA (`generate`) funciona a parte.
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
            cfg: configuración completa, se usa la clave "bev" (size, range_m) del diccionario.
        """
        b = cfg.get("bev", {})
        self.size = int(b.get("size", 128))            # Height = Width
        self.range_m = float(b.get("range_m", 50.0))   # semilado del área BEV cuadrada
        self.channels = CHANNELS
        self.n_channels = len(CHANNELS)
        self.mpp = (2.0 * self.range_m) / self.size    # metros por píxel

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
        cosinus, sinus = math.cos(yaw_rad), math.sin(yaw_rad)

        fx = dx * cosinus + dy * sinus          # eje x -> longitudinal
        fy = -dx * sinus + dy * cosinus         # eje y -> lateral
        col = self.size / 2.0 + fy / self.mpp
        row = self.size / 2.0 - fx / self.mpp   # delante = hacia arriba (row menor)

        return np.stack([col, row], axis=1)

    @staticmethod
    def box_corners(x, y, yaw_deg, ex, ey) -> np.ndarray:
        """Devuelve las 4 esquinas (mundo) de una caja centrada en (x,y), con inclinación yaw (grados) y
        la mitad de la longitud de sus lados(ex,ey)."""
        yaw_rad = math.radians(yaw_deg)
        cosinus, sinus = math.cos(yaw_rad), math.sin(yaw_rad)

        corners = np.array([[ex, ey], [ex, -ey], [-ex, -ey], [-ex, ey]], dtype=np.float32)    #  4 esquinas de un rectángulo centrado en el origen (0,0), sin rotar
        R = np.array([[cosinus, -sinus], [sinus, cosinus]], dtype=np.float32)   # matriz de rotación 2D

        corners_rotated = corners @ R.T     # rotación de las cuatro esquinas centradas en (0,0)

        return corners_rotated + np.array([x, y], dtype=np.float32)     # translación de las cuatro esquinas alrededor del punto (x,y)

    def _fill_convex(self, mask: np.ndarray, poly_px: np.ndarray, value: float = 1.0):
        """Rasteriza (rellena) un polígono convexo sobre mask, dado en píxeles (col,row).
        Es decir, pinta polígonos con coordenadas del mundo en un canal BEV que es una grid binaria.

        Test de semiplano por producto cruzado: un píxel está dentro si queda al mismo lado
        de las K aristas del polígono. Vale para cualquier orden de giro (horario/antihorario).

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

        # Producto cruzado candidato-arista (una fila por arista, vectorizado sobre los candidatos)
        K = len(poly)
        cross = np.stack([
            (poly[(i + 1) % K, 0] - poly[i, 0]) * (py - poly[i, 1]) -
            (poly[(i + 1) % K, 1] - poly[i, 1]) * (px - poly[i, 0])
            for i in range(K)
        ], axis=0)

        # Dentro del polígono = mismo signo (mismo lado) en todas las aristas a la vez
        inside = np.all(cross >= -1e-6, axis=0) | np.all(cross <= 1e-6, axis=0)
        idx = np.where(inside)[0]
        mask[py[idx].astype(int), px[idx].astype(int)] = value

