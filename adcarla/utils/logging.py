"""MetricLogger: registro de métricas de entrenamiento y evaluación en CSV (+ TensorBoard)."""
import csv
import json
import os
import time
from typing import Optional


class MetricLogger:
    """Escribe métricas por episodio en CSV y, si se puede, en TensorBoard."""

    def __init__(self, config: dict, name: Optional[str] = None, run_dir: Optional[str] = None):
        """Crea el directorio de la ejecución y vuelca el config.

        Args:
            config: configuración completa. Se usan `train.log_dir` (por defecto "runs/")
                y `train.tensorboard` (por defecto True si la librería está disponible).
            name: nombre de la ejecución; por defecto `config["name"]` o "run".
                Suele ser "teacher", "student_vision" o "student_fusion".
            run_dir: para forzar un directorio concreto (útil al reanudar un entrenamiento
                o en los tests). Si se da, no se añade timestamp.
        """
        train_config = config.get("train", {})
        self.name = name or config.get("name", "run")

        if run_dir is None:
            log_dir = train_config.get("log_dir", "runs/")
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join(log_dir, f"{self.name}_{timestamp}")
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)

        self.csv_path = os.path.join(self.run_dir, "metrics.csv")
        self._fieldnames: list = []     # columnas conocidas, en orden de aparición
        self._rows: list = []           # todas las filas escritas (para poder reescribir la cabecera)

        # Vuelca el config para que la ejecución sea reproducible sin adivinar hiperparámetros.
        with open(os.path.join(self.run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False, default=str)

        self._writer = self._make_tensorboard_writer(train_config)

        print(f"[MetricLogger] registrando en {self.run_dir}")

    def _make_tensorboard_writer(self, train_config: dict):
        """Devuelve un SummaryWriter, o None si TensorBoard está desactivado o no instalado."""
        if not train_config.get("tensorboard", True):
            return None
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            print("[MetricLogger] TensorBoard no disponible (falta tensorboard); solo se escribe CSV.")
            return None
        return SummaryWriter(log_dir=self.run_dir)

    # ---- REGISTRO ----
    def log(self, step: int, metrics: dict, prefix: str = ""):
        """Registra una fila de métricas.

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
        """Cierra el writer de TensorBoard. El CSV ya está en disco tras cada `log()`."""
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None

    # Permite usarlo con `with MetricLogger(...) as logger:` y que cierre solo si algo peta.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
