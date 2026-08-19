"""Factoría de encoders según el config."""
from .privileged import PrivilegedBEVEncoder
from .bevformer.bevformer import BEVFormerEncoder
from .fusion.fusion_encoder import FusionEncoder

def build_encoder(cfg: dict):
    name = cfg.get("encoder", "privileged")
    embed_dim = int(cfg["world_model"]["embed_dim"])
    if name == "privileged":
        return PrivilegedBEVEncoder(int(cfg["bev"]["channels"]), embed_dim, int(cfg["bev"]["size"]))
    if name == "bevformer":
        return BEVFormerEncoder(embed_dim)
    if name == "fusion":
        return FusionEncoder(embed_dim)
    raise ValueError(f"Encoder desconocido: {name}")
