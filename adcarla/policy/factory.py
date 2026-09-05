"""Construcción de Actor y Critic a partir de la configuración.

Un único sitio que traduce el bloque `policy:` del YAML a argumentos del constructor: los tres
trainers y `scripts/evaluate.py` construyen la política igual, y ningún hiperparámetro puede
quedarse en un valor por defecto del código sin que nadie lo note.
"""
from adcarla.policy.actor_critic import Actor, Critic


def build_actor(config: dict, feat_dim: int, num_actions: int) -> Actor:
    """Actor con el decay de la normalización de retornos del YAML (`policy.return_norm_decay`)."""
    return Actor(feat_dim, num_actions, float(config["policy"]["return_norm_decay"]))


def build_critic(config: dict, feat_dim: int) -> Critic:
    """Critic con el rango de bins y el decay del target lento del YAML.

    Comparte `world_model.num_bins` con la cabeza de recompensa pero NO su rango: el de la
    recompensa cubre un tick, el del critic un λ-return de cientos de ticks descontados
    (`policy.critic_bin_min`/`critic_bin_max`).
    """
    policy_config = config["policy"]
    return Critic(feat_dim, int(config["world_model"]["num_bins"]),
                  float(policy_config["critic_bin_min"]), float(policy_config["critic_bin_max"]),
                  float(policy_config["critic_ema_decay"]))
