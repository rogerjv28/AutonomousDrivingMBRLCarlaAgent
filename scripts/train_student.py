"""Etapa 2: entrena un alumno (visión o fusión) con guidance. Requiere servidor CARLA y el checkpoint del profesor.

Las opciones `init_rssm_from_teacher` e `init_actor_critic_from_teacher` siguen la misma
precedencia que `--seeds` y `--weathers` en scripts/evaluate.py: se toma el valor del YAML
(`train.init_rssm_from_teacher` y `train.init_actor_critic_from_teacher`) y la línea de comandos
solo lo sustituye si se indica de forma explícita.

Ejemplo:

    # Entrenamiento por defecto: el alumno parte de los pesos del profesor
    python scripts/train_student.py --config configs/student_vision.yaml

"""
from __future__ import annotations
import argparse
import os
from adcarla.utils.config import load_config
from adcarla.utils.seeding import set_seed
from adcarla.training.student_trainer import train_student


def resolve_init_rssm_from_teacher(config: dict, cli_value) -> bool:
    """Devuelve `train.init_rssm_from_teacher` del YAML (por defecto True), salvo que se indique por línea de comandos.

    Args:
        config: configuración completa ya cargada (con `base.yaml` incorporado).
        cli_value: `None` si no se pasó `--init-rssm-from-teacher` ni `--no-init-rssm-from-teacher`
            (entonces se usa el YAML); `True` o `False` si se pasó.
    """
    if cli_value is not None:
        return bool(cli_value)
    return bool(config.get("train", {}).get("init_rssm_from_teacher", True))


def resolve_init_actor_critic_from_teacher(config: dict, cli_value) -> bool:
    """Devuelve `train.init_actor_critic_from_teacher` del YAML (por defecto True), salvo que se indique por línea de comandos.

    Args:
        config: configuración completa ya cargada (con `base.yaml` incorporado).
        cli_value: `None` si no se pasó `--init-actor-from-teacher` ni `--no-init-actor-from-teacher`
            (entonces se usa el YAML); `True` o `False` si se pasó.
    """
    if cli_value is not None:
        return bool(cli_value)
    return bool(config.get("train", {}).get("init_actor_critic_from_teacher", True))


def main():
    p = argparse.ArgumentParser(description="Entrena un alumno con guidance (Etapa 2).")
    p.add_argument("--config", required=True, help="configs/student_vision.yaml o configs/student_fusion.yaml")
    p.add_argument("--resume", action="store_true",
                   help="Reanuda desde el último checkpoint guardado (modelo, optimizadores y episodio).")
    p.add_argument("--init-rssm-from-teacher", dest="init_rssm_from_teacher",
                   action=argparse.BooleanOptionalAction, default=None,
                   help="Preinicializa el RSSM del alumno con los pesos del profesor "
                        "(train.init_rssm_from_teacher en el YAML, por defecto true). "
                        "--no-init-rssm-from-teacher entrena el RSSM del alumno desde cero.")
    p.add_argument("--init-actor-from-teacher", dest="init_actor_critic_from_teacher",
                   action=argparse.BooleanOptionalAction, default=None,
                   help="Inicializa el actor y el critic del alumno con los del profesor, en vez de pesos "
                        "aleatorios (train.init_actor_critic_from_teacher en el YAML, por defecto true). "
                        "--no-init-actor-from-teacher entrena el actor-critic desde cero.")
    args = p.parse_args()
    if not os.path.isfile(args.config):
        p.error(f"Fichero de configuración no encontrado: {args.config}")
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 0))
    train_student(cfg, resume=args.resume,
                  init_rssm_from_teacher=resolve_init_rssm_from_teacher(cfg, args.init_rssm_from_teacher),
                  init_actor_critic_from_teacher=resolve_init_actor_critic_from_teacher(cfg, args.init_actor_critic_from_teacher))


if __name__ == "__main__":
    main()
