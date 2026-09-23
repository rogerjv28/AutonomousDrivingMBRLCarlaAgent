"""MetricLogger: registro de métricas de entrenamiento y evaluación en CSV (+ TensorBoard + log).

Produce los siguientes artefactos por ejecución, todos dentro de `runs/<name>_<time_stamp>/`:
  - `metrics.csv`: todas las métricas, una fila por episodio (para análisis).
  - `config.json`: snapshot del config completo (reproducibilidad).
  - `train.log`: una línea por episodio.
  - `bev_samples/ep_XXXX.png`: comparativa GT vs decoder cada N episodios.
"""
import csv
import json
import os
import time
from typing import Optional


class MetricLogger:
    """Escribe métricas por episodio en CSV y, si se puede, en TensorBoard."""

    def __init__(self, config: dict, name: Optional[str] = None, run_dir: Optional[str] = None,
                 total_episodes: int = 0):
        """Crea el directorio de la ejecución, vuelca el config y abre el log.

        Args:
            config: configuración completa. Se usan `train.log_dir` (por defecto "runs/"),
                `train.log_dir_human` (por defecto "logs/") y `train.tensorboard`
                (por defecto True si la librería está disponible).
            name: nombre de la ejecución; por defecto `config["name"]` o "run".
                Suele ser "teacher", "student_vision" o "student_fusion".
            run_dir: para forzar un directorio concreto (útil al reanudar un entrenamiento
                o en los tests). Si se da, no se añade timestamp.
            total_episodes: número total de episodios previstos, solo para la leyenda del log.
        """
        train_config = config.get("train", {})
        self.name = name or config.get("name", "run")
        self.total_episodes = total_episodes

        if run_dir is None:
            log_dir = train_config.get("log_dir", "runs/")
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join(log_dir, f"{self.name}_{timestamp}")
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)

        # Directorio BEV dentro del run (se crea solo cuando hay imágenes que guardar)
        self.bev_dir = os.path.join(self.run_dir, "bev_samples")

        self.csv_path = os.path.join(self.run_dir, "metrics.csv")
        self._fieldnames: list = []     # columnas conocidas, en orden de aparición
        self._rows: list = []           # todas las filas escritas (para poder reescribir la cabecera)
        # Si `run_dir` viene de un checkpoint de --resume y ya tiene un metrics.csv, lo
        # recarga para que la curva completa quede en un unico fichero (ver BaseTrainer.load_checkpoint).
        self._load_previous_csv()

        # Detectar resume: si hay métricas previas, es un resume
        is_resume = len(self._rows) > 0

        # Guarda el config: si existe y es igual, no sobrescribe; si cambió, versiona como config_N
        self._save_config(config)

        # Log: una línea por episodio, junto al CSV en el directorio de la ejecución.
        # Se puede seguir en vivo con `tail -f runs/<name>_<time_stamp>/train.log`.
        self._log_path = os.path.join(self.run_dir, "train.log")
        self._log_file = open(self._log_path, "a", encoding="utf-8", buffering=1)  # buffering=1 → flush por línea
        self._init_log_file(total_episodes, is_resume)

        self._writer = self._make_tensorboard_writer(train_config, is_resume=is_resume)

        print(f"[MetricLogger] registrando en {self.run_dir}")
        print(f"[MetricLogger] log en  {self._log_path}")

    def _make_tensorboard_writer(self, train_config: dict, is_resume: bool = False):
        """Devuelve un SummaryWriter, o None si TensorBoard está desactivado o no instalado.

        Si es resume, marca el evento de reanudación en la curva con un texto para claridad visual.
        """
        if not train_config.get("tensorboard", True):
            return None
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            print("[MetricLogger] TensorBoard no disponible (falta tensorboard); solo se escribe CSV.")
            return None

        writer = SummaryWriter(log_dir=self.run_dir)
        # Marcar visualmente en TensorBoard dónde se reanudó el entrenamiento
        if is_resume:
            writer.add_text("info/resume", f"Entrenamiento retomado {time.strftime('%Y-%m-%d %H:%M:%S')}", 0)
        return writer

    # ---- REGISTRO ----

    # Métricas que aparecen en el log y en qué orden (las que no estén se omiten sin error)
    _LOG_GROUPS = [
        # (etiqueta_grupo, [(clave_interna, etiqueta_corta, formato)])
        ("wm",     [("loss",    "loss",   "7.3f"), ("recon",   "recon",  "7.1f"),
                    ("kl",      "kl",     "5.2f"), ("reward",  "rwd",    "5.3f"),
                    ("cont",    "cnt",    "5.3f")]),
        ("policy", [("actor_loss",  "actor",  "7.4f"), ("critic_loss", "critic", "7.4f")]),
        # Las claves deben ser EXACTAMENTE las que emite DrivingMetrics.summary() (metrics.py);
        # "driving_score"/"collision_rate"/"infraction_rate" no existen ahí y no aparecían nunca.
        ("env",    [("mean_driving_score", "score", "5.3f"), ("collisions_per_episode", "col", "5.3f"),
                    ("infractions_per_episode", "inf", "5.3f"), ("mean_route_completion", "route", "5.3f")]),
    ]

    def _format_log_line(self, step: int, metrics: dict) -> str:
        """Formatea una línea de log con los grupos de métricas disponibles.

        Ejemplo de salida:
          2026-09-04 10:40:18 | ep  042/1000 | loss=  1.234 recon= 45.2 kl= 1.12 | actor=-0.0023 critic= 0.450 | score=0.673 col=0.100 | 88s
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        total = self.total_episodes or "?"
        parts = [f"{timestamp} | ep {step:4d}/{total}"]

        for _group, fields in self._LOG_GROUPS:
            tokens = []
            for key, label, fmt in fields:
                value = metrics.get(key)
                if value is not None and isinstance(value, (int, float)):
                    tokens.append(f"{label}={value:{fmt}}")
            if tokens:
                parts.append("  ".join(tokens))

        # Episode duration at the end
        seconds = metrics.get("episode_seconds")
        if seconds is not None:
            parts.append(f"{int(seconds)}s")

        return " | ".join(parts) + "\n"

    def log(self, step: int, metrics: dict, prefix: str = ""):
        """Registra una fila de métricas en CSV, TensorBoard y el log.

        Args:
            step: eje X de las curvas. En este proyecto es el número de episodio.
            metrics: dict de nombre -> valor numérico. Los valores no numéricos se
                convierten a str en el CSV y se omiten en TensorBoard.
            prefix: prefijo opcional para agrupar en TensorBoard, p. ej. "wm" o "policy"
                (se escribe como "wm/loss").
        """
        row = {"step": int(step)}
        for key, value in metrics.items():
            column = f"{prefix}_{key}" if prefix else key
            row[column] = self._to_number(value)

        self._rows.append(row)
        self._append_row(row)

        if self._writer is not None:
            for key, value in metrics.items():
                number = self._to_number(value)
                if isinstance(number, (int, float)):
                    tag = f"{prefix}/{key}" if prefix else key
                    self._writer.add_scalar(tag, number, int(step))

        # Log: una línea por episodio con las métricas principales
        try:
            self._log_file.write(self._format_log_line(step, metrics))
        except Exception:
            pass   # el log nunca debe bloquear el entrenamiento

    @staticmethod
    def _to_number(value):
        """Convierte tensores de 0 dimensiones y numéricos a float; deja el resto tal cual."""
        item = getattr(value, "item", None)   # tensores de torch y escalares de numpy
        if callable(item):
            try:
                return float(value.item())
            except (ValueError, TypeError, RuntimeError):
                return str(value)
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return value
        return str(value)

    def _load_previous_csv(self) -> None:
        """Si `self.csv_path` ya existe (reanudación en el mismo `run_dir`), recarga sus
        filas y cabecera en memoria para que las próximas llamadas a `log()` se añadan a la MISMA
        curva en vez de abrir una vacía. Si el fichero no se puede leer, falla con un
        RuntimeError explícito: mejor parar que seguir y dejar un CSV incoherente en silencio.
        """
        if not os.path.isfile(self.csv_path):
            return
        try:
            with open(self.csv_path, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames or [])
                rows = list(reader)
        except Exception as error:   # OSError, csv.Error, UnicodeDecodeError... cualquiera vale
            raise RuntimeError(
                f"No se pudo reanudar el CSV de métricas en '{self.csv_path}': {error}. "
                "Seguir habría producido una curva de entrenamiento incompleta sin avisar; "
                "borra o renombra ese metrics.csv si quieres empezar uno nuevo."
            ) from error
        if fieldnames:
            self._fieldnames = fieldnames
            self._rows = rows

    def _save_config(self, config: dict) -> None:
        """Guarda el config: si existe y es igual, no sobrescribe. Si cambió, versiona como config_1, config_2, etc.

        Permite reanudaciones (--resume) sin perder el config original, y detecta cambios de
        configuración entre reanudaciones.
        """
        config_path = os.path.join(self.run_dir, "config.json")

        # Si existe, comparar con el anterior
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    previous_config = json.load(f)
                if self._configs_equal(config, previous_config):
                    return  # Nada nuevo, no sobrescribir
            except Exception:
                pass  # Si hay error leyendo, proceder a versionar

        # Config nuevo o diferente: versionar si ya existe
        if os.path.exists(config_path):
            # Encontrar el siguiente número de versión disponible
            version = 1
            while os.path.exists(os.path.join(self.run_dir, f"config_{version}.json")):
                version += 1
            versioned_path = os.path.join(self.run_dir, f"config_{version}.json")
            with open(versioned_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False, default=str)
            print(f"[MetricLogger] config cambió entre resumptions; guardado en config_{version}.json")
        else:
            # Primera vez: guardar como config.json
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False, default=str)

    @staticmethod
    def _configs_equal(config1: dict, config2: dict, ignore_keys: set = None) -> bool:
        """Compara dos configs ignorando claves volátiles (semillas, timestamps, etc.).

        Args:
            config1, config2: configuraciones a comparar.
            ignore_keys: conjunto de claves a ignorar (por defecto: {"seed"}).

        Returns:
            True si son iguales (salvo las claves ignoradas).
        """
        import copy
        if ignore_keys is None:
            ignore_keys = {"seed"}

        config1_copy = copy.deepcopy(config1)
        config2_copy = copy.deepcopy(config2)

        # Remove ignored keys at all nesting levels (recursive)
        def remove_keys_recursive(dict_obj, keys_to_remove):
            if not isinstance(dict_obj, dict):
                return
            for key in list(dict_obj.keys()):
                if key in keys_to_remove:
                    dict_obj.pop(key)
                else:
                    remove_keys_recursive(dict_obj[key], keys_to_remove)

        remove_keys_recursive(config1_copy, ignore_keys)
        remove_keys_recursive(config2_copy, ignore_keys)

        return config1_copy == config2_copy

    def _init_log_file(self, total_episodes: int, is_resume: bool) -> None:
        """Inicializa el fichero de log, detectando si es resume o nuevo.

        Args:
            total_episodes: número total de episodios previstos.
            is_resume: True si se está reanudando un entrenamiento anterior.
        """
        if is_resume:
            # Mensaje visual de reanudación
            self._log_file.write(f"\n# {'─' * 60}\n")
            self._log_file.write(f"# Entrenamiento retomado {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            self._log_file.write(f"# {'─' * 60}\n")
        else:
            # Primera ejecución: cabecera normal
            self._log_file.write(f"# {self.name}  iniciado {time.strftime('%Y-%m-%d %H:%M:%S')}"
                                 f"  total_episodes={total_episodes}\n")
            self._log_file.write(f"# run_dir: {self.run_dir}\n")
            self._log_file.write("#\n")

    def _append_row(self, row: dict):
        """Añade la fila al CSV; si aparecen columnas nuevas, reescribe el fichero entero.

        Reescribir es barato (un fichero por ejecución, del orden de miles de filas) y evita
        el problema clásico de que un `print` con claves distintas descoloque las columnas.
        """
        new_columns = [key for key in row if key not in self._fieldnames]
        if new_columns:
            self._fieldnames.extend(new_columns)
            self._rewrite()
            return

        with open(self.csv_path, "a", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction="ignore").writerow(row)

    def _rewrite(self):
        """Reescribe el CSV completo con la cabecera actual (rellena con "" lo que falte)."""
        with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames, restval="", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._rows)

    # ---- VISUALIZACIÓN BEV ----
    def save_bev(self, gt_bev, pred_logits, episode: int, channel_names: list):
        """Guarda una comparativa GT vs decoder como PNG en `runs/<run>/bev_samples/`.

        El decoder devuelve logits crudos y esta función aplica sigmoid internamente para
        obtener probabilidades antes de componer la imagen en color.

        Args:
            gt_bev: tensor [C, H, W] binario (la máscara privilegiada real).
            pred_logits: tensor [C, H, W] logits del decoder (sin sigmoid).
            episode: número de episodio, para el nombre del fichero y la leyenda.
            channel_names: lista con los C nombres de canal, en el mismo orden que la máscara.
        """
        try:
            from adcarla.utils.viz import save_bev_comparison
            os.makedirs(self.bev_dir, exist_ok=True)
            path = os.path.join(self.bev_dir, f"ep_{episode:04d}.png")
            save_bev_comparison(gt_bev, pred_logits, path, channel_names, episode=episode)
        except Exception as exception:
            # La visualización es opcional: no debe interrumpir el entrenamiento
            print(f"[MetricLogger] advertencia: no se pudo guardar BEV ep {episode}: {exception}")

    def log_summary(self, summary: dict, filename: str = "summary.json"):
        """Guarda un resumen final (p. ej. `DrivingMetrics.summary()`) como JSON.

        Args:
            summary: dict serializable con los resultados agregados.
            filename: nombre del fichero dentro del directorio de la ejecución.
        """
        path = os.path.join(self.run_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        print(f"[MetricLogger] resumen guardado en {path}")

    def checkpoint_path(self, filename: str) -> str:
        """Ruta dentro del directorio de la ejecución, para guardar checkpoints junto a las métricas.

        Crea el subdirectorio si hace falta. Útil para no pisar `checkpoints/teacher.pt`
        entre ejecuciones distintas.
        """
        directory = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, filename)

    def close(self):
        """Cierra el writer de TensorBoard y el log. El CSV ya está en disco tras cada `log()`."""
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None
        if self._log_file is not None and not self._log_file.closed:
            self._log_file.write(f"# finalizado {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            self._log_file.close()
            self._log_file = None

    # Permite usarlo con `with MetricLogger(...) as logger:` y que cierre solo si algo peta.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
