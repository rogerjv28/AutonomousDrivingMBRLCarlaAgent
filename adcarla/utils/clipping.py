"""Recorte de gradientes: por norma global (el de siempre) y AGC adaptativo (DreamerV3)."""

import torch


def unit_norm(tensor: torch.Tensor) -> torch.Tensor:
    """Norma L2 por unidad del tensor, conservando las dimensiones para poder difundir.

    Unidad = fila en un peso [salida, entrada], filtro en una conv [out, in, kh, kw]. Los tensores
    de una sola dimensión (sesgos, escalas de RMSNorm) cuentan como una unidad entera.
    """
    if tensor.ndim <= 1:
        return tensor.norm(2)

    return tensor.norm(2, dim=tuple(range(1, tensor.ndim)), keepdim=True)


@torch.no_grad()
def adaptive_grad_clip_(parameters, clip_factor: float = 0.3, eps: float = 1e-3) -> None:
    """Adaptive Gradient Clipping (Brock et al., ICML 2021), el que usa DreamerV3.

    Recorta el gradiente de cada unidad si supera `clip_factor` veces la norma de su peso, en vez
    de contrastar la norma global del modelo contra un umbral absoluto. El umbral deja de depender
    de la escala de la pérdida: tocar un peso de la pérdida ya no obliga a reajustar el recorte.

    Args:
        parameters: iterable de parámetros (los que no tienen gradiente se ignoran).
        clip_factor: fracción de la norma del peso que puede alcanzar su gradiente (0.3 en
            DreamerV3, que lo describe como "30% of the L2 norm of the weight matrix").
        eps: suelo de la norma del peso. Sin él, un tensor inicializado a cero (los sesgos lo
            están) tendría umbral cero y su gradiente se anularía para siempre.
    """
    for parameter in parameters:
        if parameter.grad is None:
            continue

        weight_norm_threshold  = unit_norm(parameter.detach()).clamp(min=eps) * clip_factor
        grad_norm  = unit_norm(parameter.grad.detach())
        clipped_grad  = parameter.grad * (weight_norm_threshold  / grad_norm .clamp(min=1e-6))
        parameter.grad.copy_(torch.where(grad_norm > weight_norm_threshold , clipped_grad , parameter.grad))
