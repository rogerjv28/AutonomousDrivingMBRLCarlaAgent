"""Etapa 1: entrena el profesor privilegiado. Requiere servidor CARLA."""
from __future__ import annotations
import argparse
import csv
import os
import statistics

from adcarla.utils.config import load_config
from adcarla.utils.seeding import set_seed
from adcarla.training.teacher_trainer import TeacherTrainer, train_teacher


def main():
    p = argparse.ArgumentParser(
        description="Entrena el profesor privilegiado (Etapa 1).",
        epilog="Ejemplos:\n"
               "  python scripts/train_teacher.py\n"
               "  python scripts/train_teacher.py --time-episodes 20\n"
               "  python scripts/train_teacher.py train.total_episodes=500 train.train_ratio=8\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default="configs/teacher.yaml",
                   help="Ruta al fichero YAML del profesor (por defecto: configs/teacher.yaml).")
    p.add_argument("--resume", action="store_true",
                   help="Reanuda desde el último checkpoint guardado (modelo, optimizadores y episodio).")
    p.add_argument("--time-episodes", type=int, default=0, metavar="N",
                   help="Modo cronómetro: ejecuta N episodios, imprime estadísticas de tiempo y termina. "
                        "Se recomiendan 20 episodios (el primero se descarta porque incluye la carga inicial). "
                        "Sirve para estimar la duración del entrenamiento completo antes de alquilar GPU.")
    p.add_argument("overrides", nargs="*", metavar="CLAVE=VALOR",
                   help="Sobreescribe valores del YAML sin editar el fichero. "
                        "CLAVE puede indicar una clave anidada con puntos (train.total_episodes). "
                        "El tipo se toma del valor que ya tiene la clave en el YAML. "
                        "Ejemplo: train.total_episodes=500 train.train_ratio=8")
    args = p.parse_args()

    if not os.path.isfile(args.config):
        p.error(f"Fichero de configuración no encontrado: {args.config}")

    cfg = load_config(args.config)
    _apply_overrides(cfg, args.overrides)
    set_seed(cfg.get("seed", 0))

    if args.time_episodes:
        _run_timing_mode(cfg, args.time_episodes)
    else:
        train_teacher(cfg, resume=args.resume)


# Sobreescritura de la configuración desde la línea de órdenes

def _apply_overrides(cfg: dict, overrides: list) -> None:
    """Aplica al config las sobreescrituras CLAVE=VALOR. CLAVE puede ser anidada (train.total_episodes)."""
    for token in overrides:
        if "=" not in token:
            raise SystemExit(
                f"Override mal formado (falta '='): {token!r}\n"
                f"Usa la forma CLAVE=VALOR, p. ej. train.total_episodes=500"
            )
        key, raw_value = token.split("=", 1)
        parts = key.split(".")
        node = cfg
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise SystemExit(
                    f"Override: la clave intermedia '{'.'.join(parts[:-1])}' no existe en el config."
                )
            node = node[part]
        leaf = parts[-1]
        if leaf in node:
            existing = node[leaf]
            try:
                if isinstance(existing, bool):
                    node[leaf] = raw_value.lower() in ("1", "true", "yes")
                elif isinstance(existing, int):
                    node[leaf] = int(raw_value)
                elif isinstance(existing, float):
                    node[leaf] = float(raw_value)
                else:
                    node[leaf] = raw_value
            except ValueError:
                node[leaf] = raw_value
        else:
            # Clave nueva: no hay tipo de referencia, se guarda como texto.
            node[leaf] = raw_value


# Modo cronómetro

def _run_timing_mode(cfg: dict, n_episodes: int) -> None:
    """Ejecuta N episodios y muestra estadísticas de tiempo por episodio.

    Fija total_episodes en N y prefill_episodes en 0, para que solo se midan episodios
    de entrenamiento y no los de exploración aleatoria.
    """
    cfg = dict(cfg)
    cfg["train"] = dict(cfg.get("train", {}))
    cfg["train"]["total_episodes"] = n_episodes
    cfg["train"]["prefill_episodes"] = 0   # el prefill no se mide

    print(f"\n[cronómetro] Ejecutando {n_episodes} episodios de profesor para medir tiempos…\n")
    trainer = TeacherTrainer(cfg)
    trainer.train()

    # Lee episode_seconds del CSV que escribe el logger
    csv_path = trainer.logger.csv_path
    tiempos = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for fila in csv.DictReader(f):
                val = fila.get("episode_seconds", "").strip()
                if val:
                    tiempos.append(float(val))
    except FileNotFoundError:
        print(f"[cronómetro] No se encontró el CSV en {csv_path}.")
        return
    except ValueError as e:
        print(f"[cronómetro] Error: valor no numérico en episode_seconds: {e}")
        return

    if not tiempos:
        print("[cronómetro] La columna 'episode_seconds' no aparece en el CSV.")
        return

    # Se descarta el primer episodio porque incluye la carga del entorno y del modelo
    muestra = tiempos[1:] if len(tiempos) > 1 else tiempos
    media = statistics.mean(muestra)
    desv = statistics.stdev(muestra) if len(muestra) > 1 else 0.0

    print()
    print("=" * 62)
    print(f"  CRONÓMETRO: {n_episodes} episodios completados")
    print("=" * 62)
    print(f"  Episodios medidos (sin el 1.º de carga): {len(muestra)}")
    print(f"  Tiempo por episodio:  {media:.1f} ± {desv:.1f} s")
    print()
    print("  Estimación para el entrenamiento del profesor (sin los alumnos):")
    print()
    for total_ep in [500, 800, 1_000, 1_500, 5_000]:
        horas = media * total_ep / 3600
        dias = horas / 24
        print(f"    total_episodes={total_ep:5d}  =  {horas:6.1f} h  ({dias:.1f} días)")
    print()
    print("  Nota: cada alumno tarda más por episodio que el profesor, porque calcula")
    print("  también los términos de guidance y hace correr al profesor en paralelo.")
    print("=" * 62)
    print(f"\n  CSV completo: {csv_path}")


if __name__ == "__main__":
    main()
