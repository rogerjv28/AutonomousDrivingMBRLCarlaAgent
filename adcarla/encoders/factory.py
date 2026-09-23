"""Factoría de encoders según el config.

Además de instanciar, valida que el config no prometa componentes que no existen.
"""
from .base import BEVEncoder
from .privileged import PRIVILEGED_REDUCTION, PrivilegedBEVEncoder
from .bevformer.bevformer import BEVFormerEncoder
from .fusion.fusion_encoder import FusionEncoder

SUPPORTED_BACKBONES = {"cnn_light"}
SUPPORTED_LIDAR_BRANCHES = {"rasterized_cnn"}
SUPPORTED_FUSERS = {"conv"}


def _camera_params(config: dict) -> dict:
    """Lee y valida el bloque `bevformer:` del config. Devuelve los kwargs del encoder de cámara.

    Raises:
        ValueError: si el config pide un backbone o una fusión temporal no implementados.
    """
    camera_cfg = config.get("bevformer", {}) or {}

    backbone = str(camera_cfg.get("backbone", "cnn_light"))
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"bevformer.backbone = '{backbone}' no está implementado. Valores válidos: "
            f"{sorted(SUPPORTED_BACKBONES)}. El backbone actual es una CNN propia de 3 capas."
        )
    if camera_cfg.get("temporal", False):
        raise ValueError(
            "bevformer.temporal = true, pero no hay módulo de temporal self-attention "
            "implementado. Ponlo a false o implementa encoders/bevformer/temporal_self_attn.py."
        )

    return {
        "bev_channels": int(camera_cfg.get("bev_channels", 128)),
        "num_heads": int(camera_cfg.get("num_heads", 4)),
        "backbone_channels": int(camera_cfg.get("backbone_channels", 64)),
    }


def _fusion_params(config: dict) -> dict:
    """Lee y valida el bloque `fusion:` del config. Devuelve los kwargs propios de la fusión.

    Raises:
        ValueError: si el config pide una rama de LiDAR o un fusor no implementados.
    """
    fusion_cfg = config.get("fusion", {}) or {}

    lidar_branch = str(fusion_cfg.get("lidar", "rasterized_cnn"))
    if lidar_branch not in SUPPORTED_LIDAR_BRANCHES:
        raise ValueError(
            f"fusion.lidar = '{lidar_branch}' no está implementado. Valores válidos: "
            f"{sorted(SUPPORTED_LIDAR_BRANCHES)}. La rama actual rasteriza el LiDAR a "
            f"ocupación+altura y le pasa una CNN de 2 capas."
        )

    fuser = str(fusion_cfg.get("fuser", "conv"))
    if fuser not in SUPPORTED_FUSERS:
        raise ValueError(
            f"fusion.fuser = '{fuser}' no está implementado. Valores válidos: {sorted(SUPPORTED_FUSERS)}."
        )

    return {"lidar_in_channels": int(fusion_cfg.get("lidar_in_channels", 2))}


def build_encoder(config: dict) -> BEVEncoder:
    """Instancia el encoder de sensores según config["encoder"].

    Args:
        config: configuración completa. Debe contener 'world_model.embed_dim' y 'bev.grid_size';
            para el encoder privilegiado también 'bev.channels' y 'bev.size'. Los bloques
            'bevformer' y 'fusion' son opcionales (se usan los valores por defecto del código).

    Returns:
        Encoder instanciado: PrivilegedBEVEncoder, BEVFormerEncoder o FusionEncoder.

    Raises:
        RuntimeError: si faltan claves necesarias en config.
        ValueError: si el nombre del encoder no es reconocido o el config pide componentes
            no implementados.
    """
    name = config.get("encoder", "privileged")
    if "world_model" not in config or "embed_dim" not in config.get("world_model", {}):
        raise RuntimeError("Falta la clave 'world_model.embed_dim' en la configuración del encoder")
    embed_dim = int(config["world_model"]["embed_dim"])

    # `bev.grid_size` es compartido por los tres encoders a propósito.
    bev_cfg = config.get("bev", {}) or {}
    if "grid_size" not in bev_cfg:
        raise RuntimeError("Falta la clave 'bev.grid_size' (lado de la rejilla BEV) en la configuración")
    grid_size = int(bev_cfg["grid_size"])

    if name == "privileged":
        if "channels" not in bev_cfg or "size" not in bev_cfg:
            raise RuntimeError("Falta 'bev.channels' o 'bev.size' en la configuración para el encoder privilegiado")
        # El profesor reduce la máscara x8 (PRIVILEGED_REDUCTION) antes de la rejilla.
        reduced_bev_size = int(bev_cfg["size"]) // PRIVILEGED_REDUCTION
        if reduced_bev_size < grid_size:
            raise RuntimeError(
                f"bev.size = {bev_cfg['size']} reduce a {reduced_bev_size}x{reduced_bev_size} en el encoder "
                f"privilegiado (÷{PRIVILEGED_REDUCTION}), por debajo de bev.grid_size = {grid_size}: "
                f"la rejilla del profesor saldría ampliada. Sube bev.size a "
                f"{grid_size * PRIVILEGED_REDUCTION} o baja bev.grid_size.")
        return PrivilegedBEVEncoder(int(bev_cfg["channels"]), embed_dim, grid_size)

    if name == "bevformer":
        return BEVFormerEncoder(embed_dim, grid_size=grid_size, **_camera_params(config))

    if name == "fusion":
        # La rama de fusión reutiliza la misma arquitectura de cámara que la rama visión
        # (pesos independientes), de ahí que comparta _camera_params().
        return FusionEncoder(embed_dim, grid_size=grid_size, **_camera_params(config), **_fusion_params(config))

    raise ValueError(f"Encoder desconocido: '{name}'. Valores válidos: 'privileged', 'bevformer', 'fusion'")
