"""Visualización del mapa BEV: composición semántica en color y comparativa GT vs decoder."""
import os

import cv2
import numpy as np
import torch

# Un color RGB por canal semántico (orden de pintado: los últimos tapan a los anteriores)
_PALETTE = {
    "road":        (100, 100, 100),   # gris
    "route":       (255, 200,   0),   # amarillo
    "vehicle":     (220,  40,  40),   # rojo
    "pedestrian":  (255, 140,   0),   # naranja
    "light_green": ( 50, 200,  50),   # verde
    "light_red":   (220,  20,  60),   # carmesí
    "ego":         ( 30, 120, 255),   # azul (encima de todo para que el ego siempre sea visible)
}

# Orden de composición: el ego va siempre al frente, la carretera de fondo
_DRAW_ORDER = ["road", "route", "vehicle", "pedestrian", "light_green", "light_red", "ego"]


def bev_to_rgb(mask: np.ndarray, channel_names: list) -> np.ndarray:
    """Convierte una máscara BEV [C, H, W] en imagen RGB [H, W, 3].

    Cada canal se compone con su color semántico usando blending lineal, en el orden
    de `_DRAW_ORDER` para que el ego quede siempre encima. Canales no listados en
    la paleta se renderizan en gris claro para no perder información.

    Args:
        mask: array float32 [C, H, W] con valores en [0, 1] — binario para el GT,
              probabilidades (sigmoid de logits) para la predicción del decoder.
        channel_names: lista con los nombres de los C canales, en el mismo orden que la máscara.

    Returns:
        Imagen RGB [H, W, 3] uint8 sobre fondo negro.
    """
    C, H, W = mask.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    name_to_idx = {name: i for i, name in enumerate(channel_names)}

    # Primero los canales conocidos, en orden de prioridad visual
    rendered = set()
    for ch_name in _DRAW_ORDER:
        if ch_name not in name_to_idx:
            continue
        idx = name_to_idx[ch_name]
        if idx >= C:
            continue
        alpha = mask[idx, :, :][:, :, np.newaxis]          # [H, W, 1]
        color = np.array(_PALETTE[ch_name], dtype=np.float32)
        rgb = (1.0 - alpha) * rgb + alpha * color
        rendered.add(ch_name)

    # Canales que no tienen color definido: gris claro para no ocultarlos
    for ch_name, idx in name_to_idx.items():
        if ch_name in rendered or idx >= C:
            continue
        alpha = mask[idx, :, :][:, :, np.newaxis]
        rgb = (1.0 - alpha) * rgb + alpha * np.array([180, 180, 180], dtype=np.float32)

    return rgb.clip(0, 255).astype(np.uint8)


def save_bev_comparison(
    gt_bev: torch.Tensor,
    pred_logits: torch.Tensor,
    path: str,
    channel_names: list,
    episode: int = -1,
):
    """Guarda una imagen PNG con GT (izquierda) y la predicción del decoder (derecha).

    Aplica sigmoid a `pred_logits` internamente: el decoder devuelve logits crudos y
    esta función es el único sitio fuera del loss donde se necesita la probabilidad.

    Args:
        gt_bev: tensor [C, H, W] binario (0/1), la máscara privilegiada real.
        pred_logits: tensor [C, H, W] con los logits crudos del decoder (sin sigmoid).
        path: ruta completa del fichero PNG a guardar (el directorio se crea si no existe).
        channel_names: lista con los nombres de los C canales.
        episode: número de episodio, solo para la leyenda de la imagen.
    """
    gt_np = gt_bev.detach().cpu().float().numpy()
    pred_prob = torch.sigmoid(pred_logits.detach().cpu().float()).numpy()

    gt_rgb = bev_to_rgb(gt_np, channel_names)
    pred_rgb = bev_to_rgb(pred_prob, channel_names)

    H, W = gt_rgb.shape[:2]
    gap = np.full((H, 6, 3), 40, dtype=np.uint8)   # separador gris oscuro de 6 px
    composite = np.concatenate([gt_rgb, gap, pred_rgb], axis=1)

    # Banda de título de 18 px en la parte superior
    title_band = np.full((18, composite.shape[1], 3), 25, dtype=np.uint8)
    composite = np.concatenate([title_band, composite], axis=0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    ep_txt = f"ep {episode:04d}" if episode >= 0 else ""
    cv2.putText(composite, f"GT privilegiado  {ep_txt}", (4, 13),
                font, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(composite, "Decoder (prediccion)", (W + 10, 13),
                font, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
