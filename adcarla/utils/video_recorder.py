"""VideoRecorder: graba vídeos MP4 de episodios CARLA con overlays opcionales de BEV.

Funciona sin GUI y sin ffmpeg: usa `cv2.VideoWriter`, ya dependencia del proyecto. Modos de uso,
layout del frame y conversión a H.264.

Uso habitual:

    with VideoRecorder("videos/run.mp4", show_bev=True, show_decoder=True) as rec:
        obs = env.reset()
        while not done:
            action = agent.act(obs, greedy=True)
            obs, reward, done, _ = env.step(action)
            rec.add_frame(
                rgb=cam.get_frame(),
                bev_gt=torch.from_numpy(obs["bev_privileged"]) if obs.get("bev_privileged") is not None else None,
                decoder_logits=decoder_pred,   # tensor [C, H, W], logits sin sigmoid
                channel_names=CHANNELS,
                info={"episode": ep, "step": t, "speed": speed, "reward": reward},
            )
"""
import os
import threading
import time
from typing import Optional

import cv2
import numpy as np
import torch

from adcarla.utils.viz import bev_to_rgb

# ---- Constantes de layout ----
_VIDEO_W = 1280
_VIDEO_H = 720
_INFO_H = 36         # banda de texto en la parte superior del frame
_CONTENT_H = _VIDEO_H - _INFO_H   # 684 px
_PANEL_W = 240       # ancho de cada panel BEV lateral
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_INFO_BG = (18, 18, 18)       # casi negro
_SEPARATOR = (55, 55, 55)     # gris oscuro para la línea divisoria


class VideoRecorder:
    """Graba un vídeo MP4 de episodios CARLA con overlays opcionales de BEV.

    Los paneles BEV (GT y/o decoder) aparecen en el lado derecho del frame; la cámara RGB del ego
    se escala para ocupar el espacio restante. Usa ``cv2.VideoWriter`` con el codec ``mp4v``;
    conversión a H.264.
    """

    def __init__(
        self,
        output_path: str,
        fps: int = 10,
        show_bev: bool = False,
        show_decoder: bool = False,
    ):
        """
        Args:
            output_path: ruta del fichero MP4 de salida. Se crea el directorio si no existe.
            fps: fotogramas por segundo. Usar 10 para tiempo real (CARLA a 10 Hz).
            show_bev: añade un panel con el BEV GT privilegiado (panel superior derecho).
            show_decoder: añade un panel con la predicción del decoder (panel inferior derecho).
        """
        self.output_path = output_path
        self.fps = fps
        self.show_bev = show_bev
        self.show_decoder = show_decoder

        self._num_panels = int(show_bev) + int(show_decoder)
        # El área RGB se estrecha para dejar sitio a los paneles BEV
        self._rgb_w = _VIDEO_W - _PANEL_W if self._num_panels > 0 else _VIDEO_W
        self._rgb_h = _CONTENT_H
        # Con 2 paneles cada uno ocupa la mitad vertical, con 1 panel ocupa todo
        self._panel_h = _CONTENT_H // 2 if self._num_panels == 2 else _CONTENT_H

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(output_path, fourcc, fps, (_VIDEO_W, _VIDEO_H))
        if not self._writer.isOpened():
            raise RuntimeError(
                f"[VideoRecorder] no se pudo abrir el writer para '{output_path}'. "
                "Comprueba que el directorio existe y que OpenCV tiene soporte de vídeo."
            )

        self._frame_count = 0
        self._t_start = time.monotonic()
        print(f"[VideoRecorder] grabando en  {output_path}  ({_VIDEO_W}×{_VIDEO_H} @ {fps} fps)")

    # ---- API pública ----

    def add_frame(
        self,
        rgb: np.ndarray,
        bev_gt: Optional[torch.Tensor] = None,
        decoder_logits: Optional[torch.Tensor] = None,
        channel_names: Optional[list] = None,
        info: Optional[dict] = None,
    ) -> None:
        """Añade un fotograma al vídeo.

        Args:
            rgb: imagen RGB del ego [H, W, 3] uint8 (espacio RGB, no BGR).
                Se escala internamente al tamaño del frame de vídeo.
            bev_gt: máscara BEV privilegiada [C, H, W] como tensor o array.
                Esperada binaria (0/1). Si show_bev=True y es None, el panel
                aparece en negro (útil cuando la máscara aún no está disponible).
            decoder_logits: logits crudos del decoder [C, H, W] como tensor.
                Se aplica sigmoid internamente, igual que en MetricLogger.save_bev.
                Si show_decoder=True y es None, el panel aparece en negro.
            channel_names: nombres de los C canales BEV en el mismo orden que las máscaras.
                Necesario para la paleta semántica de colores; si es None todos los
                canales se pintan en gris claro.
            info: dict con métricas del paso para la barra superior del frame.
                Claves reconocidas: "episode"/"ep", "step", "speed" (m/s), "reward".
        """
        frame = self._compose_frame(
            rgb, bev_gt, decoder_logits, channel_names or [], info or {}
        )
        self._writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        self._frame_count += 1

    def close(self) -> None:
        """Finaliza el fichero de vídeo y libera el writer de OpenCV."""
        if self._writer is not None and self._writer.isOpened():
            self._writer.release()
        dur = time.monotonic() - self._t_start
        print(
            f"[VideoRecorder] {self._frame_count} frames  |  {dur:.1f} s  →  {self.output_path}"
        )
        self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
        return False

    # ---- Composición del frame ----

    def _compose_frame(
        self,
        rgb: np.ndarray,
        bev_gt: Optional[torch.Tensor],
        decoder_logits: Optional[torch.Tensor],
        channel_names: list,
        info: dict,
    ) -> np.ndarray:
        """Compone el canvas completo: barra de info + RGB principal + paneles BEV."""
        canvas = np.zeros((_VIDEO_H, _VIDEO_W, 3), dtype=np.uint8)

        # ---- Barra de info (parte superior) ----
        canvas[:_INFO_H] = _INFO_BG
        self._draw_info_bar(canvas, info)

        # ---- Área RGB principal ----
        if rgb is not None and rgb.size > 0:
            rgb_scaled = cv2.resize(rgb, (self._rgb_w, self._rgb_h))
        else:
            rgb_scaled = np.zeros((self._rgb_h, self._rgb_w, 3), dtype=np.uint8)
        canvas[_INFO_H: _INFO_H + self._rgb_h, : self._rgb_w] = rgb_scaled

        # ---- Paneles BEV (lado derecho) ----
        if self._num_panels > 0:
            x0 = self._rgb_w
            panel_idx = 0

            if self.show_bev:
                y0 = _INFO_H + panel_idx * self._panel_h
                panel_img = self._tensor_to_bev_rgb(bev_gt, channel_names)
                canvas[y0: y0 + self._panel_h, x0: x0 + _PANEL_W] = cv2.resize(
                    panel_img, (_PANEL_W, self._panel_h)
                )
                cv2.putText(
                    canvas, "BEV GT", (x0 + 6, y0 + 14),
                    _FONT, 0.38, (220, 220, 80), 1, cv2.LINE_AA,
                )
                panel_idx += 1

            if self.show_decoder:
                y0 = _INFO_H + panel_idx * self._panel_h
                panel_img = self._decoder_to_rgb(decoder_logits, channel_names)
                canvas[y0: y0 + self._panel_h, x0: x0 + _PANEL_W] = cv2.resize(
                    panel_img, (_PANEL_W, self._panel_h)
                )
                cv2.putText(
                    canvas, "Decoder", (x0 + 6, y0 + 14),
                    _FONT, 0.38, (80, 200, 220), 1, cv2.LINE_AA,
                )

            # Separadores entre áreas
            canvas[_INFO_H:, x0: x0 + 2] = _SEPARATOR   # línea vertical
            if self._num_panels == 2:                   # línea entre paneles
                sep_y = _INFO_H + self._panel_h
                canvas[sep_y: sep_y + 2, x0:] = _SEPARATOR

        return canvas

    def _tensor_to_bev_rgb(
        self, tensor: Optional[torch.Tensor], channel_names: list
    ) -> np.ndarray:
        """Máscara BEV GT → imagen RGB; negro si tensor es None."""
        if tensor is None:
            return np.zeros((_PANEL_W, _PANEL_W, 3), dtype=np.uint8)
        arr = (
            tensor.detach().cpu().float().numpy()
            if isinstance(tensor, torch.Tensor)
            else np.asarray(tensor, dtype=np.float32)
        )
        return bev_to_rgb(arr, channel_names)

    def _decoder_to_rgb(
        self, logits: Optional[torch.Tensor], channel_names: list
    ) -> np.ndarray:
        """Logits del decoder → sigmoid → imagen RGB, negro si logits es None."""
        if logits is None:
            return np.zeros((_PANEL_W, _PANEL_W, 3), dtype=np.uint8)
        probs = torch.sigmoid(logits.detach().cpu().float()).numpy()
        return bev_to_rgb(probs, channel_names)

    def _draw_info_bar(self, canvas: np.ndarray, info: dict) -> None:
        """Escribe la barra superior con dos filas.

        Fila 1 (y≈13): episodio, paso, velocidad, recompensa + timestamp a la derecha.
        Fila 2 (y≈27): clima y escenario (solo si están en `info`).
        """
        ep = info.get("episode", info.get("ep", "—"))
        step = info.get("step", "—")
        speed = info.get("speed")
        reward = info.get("reward")

        # ---- Fila 1: métricas numéricas ----
        metrics_text = [f"ep {ep}  paso {step}"]
        if speed is not None:
            metrics_text.append(f"v={float(speed):.1f} m/s")
        if reward is not None:
            metrics_text.append(f"r={float(reward):+.2f}")
        cv2.putText(
            canvas, "    ".join(metrics_text), (8, 13),
            _FONT, 0.38, (200, 200, 200), 1, cv2.LINE_AA,
        )
        # Timestamp + contador de frame en la esquina derecha (misma fila)
        ts = time.strftime("%H:%M:%S")
        cv2.putText(
            canvas, f"{ts}  f{self._frame_count}", (_VIDEO_W - 150, 13),
            _FONT, 0.33, (120, 120, 120), 1, cv2.LINE_AA,
        )

        # ---- Fila 2: clima y escenario (opcionales) ----
        weather = info.get("weather")
        scenario = info.get("scenario")
        info_text = []
        if weather:
            info_text.append(str(weather))
        if scenario:
            info_text.append(str(scenario))
        if info_text:
            cv2.putText(
                canvas, "  ·  ".join(info_text), (8, 27),
                _FONT, 0.33, (140, 160, 140), 1, cv2.LINE_AA,
            )


# ---------------------------------------------------------------------------
# Cámara RGB auxiliar para grabación (independiente del encoder)
# ---------------------------------------------------------------------------

# Resolución por defecto de la cámara de grabación.
# Es independiente de la resolución del encoder (sensors.image_size).
VIDEO_CAM_W: int = 800
VIDEO_CAM_H: int = 450


class _VideoCamera:
    """Cámara RGB auxiliar montada sobre el ego para grabación.

    No interfiere con las cámaras del encoder: se spawnea sobre el mismo actor
    ego pero como sensor independiente.  Se crea justo después de cada
    ``env.reset()`` (que destruye y recrea el ego) y se destruye antes del
    siguiente reset.  Thread-safe: el callback de CARLA llega desde un hilo
    interno del cliente; ``get_frame()`` protege el acceso con un lock.

    ``import carla`` se hace en ``__init__`` para no imponer una dependencia de
    instalación del módulo carla en entornos que solo ejecutan tests.
    """

    def __init__(self, world, ego, width: int = VIDEO_CAM_W, height: int = VIDEO_CAM_H):
        """Spawnea la cámara y comienza a escuchar callbacks.

        Args:
            world: ``carla.World`` donde spawnear el sensor.
            ego: actor vehículo al que adjuntar la cámara.
            width, height: resolución de captura.  ``VideoRecorder`` escala
                internamente, así que no tiene por qué coincidir con el frame
                final del vídeo.
        """
        import carla  # importación diferida: no obligatoria en tests

        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()

        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(width))
        bp.set_attribute("image_size_y", str(height))
        bp.set_attribute("fov", "90")

        # Cámara en tercera persona: detrás y por encima del vehículo, mirando hacia adelante
        # con inclinación hacia abajo para ver el entorno además del propio coche.
        transform = carla.Transform(
            carla.Location(x=-7.0, y=0.0, z=3.0),
            carla.Rotation(pitch=-12.0, yaw=0.0, roll=0.0)
        )
        self._actor = world.spawn_actor(bp, transform, attach_to=ego)
        self._actor.listen(self._callback)

    def _callback(self, image) -> None:
        """Decodifica el frame BGRA de CARLA a RGB numpy y lo almacena."""
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
        rgb = arr[:, :, ::-1].copy()   # BGR → RGB
        with self._lock:
            self._frame = rgb

    def get_frame(self) -> Optional[np.ndarray]:
        """Devuelve el último frame RGB capturado, o None si aún no ha llegado ninguno."""
        with self._lock:
            return self._frame

    def destroy(self) -> None:
        """Detiene el listener y destruye el actor sensor en CARLA."""
        try:
            self._actor.stop()
        except Exception:
            pass
        try:
            self._actor.destroy()
        except Exception:
            pass
