"""Carga de configuración YAML con soporte de `defaults: otro.yaml` (merge)."""
import os
import yaml


def load_config(path: str) -> dict:
    """Carga un YAML; si tiene la clave `defaults`, funde primero ese fichero base."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base_name = cfg.pop("defaults", None)
    if base_name:
        base_path = os.path.join(os.path.dirname(path), base_name)
        base = load_config(base_path)
        cfg = _deep_merge(base, cfg)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
