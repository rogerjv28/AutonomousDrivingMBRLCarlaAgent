"""SequenceReplayBuffer: almacenamiento episódico y muestreo de secuencias de longitud fija.

Guarda episodios completos y muestrea ventanas contiguas de longitud seq_len, necesarias para
que el RSSM aprenda transiciones temporales. Cuando el total de steps supera `capacity` se
descartan los episodios más antiguos completos (FIFO, sin partir episodios a la mitad).
"""
import random
import numpy as np
import torch


def normalize_sensor_value(key: str, tensor: torch.Tensor) -> torch.Tensor:
    """Normalización de sensores compartida entre `sample()` y `RolloutAgent._batchify`.
    `cameras` se reescala de uint8[0,255] a float32 [0,1]. El resto de claves no cambia.
    """
    if key == "cameras":
        return tensor / 255.0
    return tensor


class SequenceReplayBuffer:
    """Buffer de observación episódica para entrenamiento del World Model con secuencias temporales."""

    def __init__(self, capacity: int, seq_len: int):
        """
        Args:
            capacity: número máximo de steps totales almacenados (suma de todos los episodios).
            seq_len: longitud mínima para que un episodio sea muestreable y longitud exacta
                     de las secuencias devueltas por sample().
        """
        self.capacity = capacity
        self.seq_len = seq_len
        self.episodes = []              # lista de episodios, cada episodio es una lista de dicts de steps
        self._current_episode = []      # episodio en curso (se cierra con end_episode)
        self._total_steps = 0           # contador de steps almacenados en todos los episodios

    def add_step(self, step: dict):
        """Añade un step al episodio en curso. Los valores None se ignoran."""
        self._current_episode.append({k: np.asarray(v) for k, v in step.items() if v is not None})

    def end_episode(self):
        """Cierra el episodio en curso y lo añade al buffer si tiene longitud suficiente.

        Descarta episodios más cortos que seq_len. Si se supera capacity, elimina episodios
        antiguos (FIFO) hasta volver a estar dentro del límite.
        """
        if len(self._current_episode) >= self.seq_len:
            self.episodes.append(self._current_episode)
            self._total_steps += len(self._current_episode)
            while self._total_steps > self.capacity and self.episodes:
                self._total_steps -= len(self.episodes.pop(0))
        self._current_episode = []

    def __len__(self) -> int:
        return self._total_steps

    def can_sample(self) -> bool:
        """True si hay al menos un episodio muestreable (longitud >= seq_len)."""
        return any(len(episode) >= self.seq_len for episode in self.episodes)

    def sample(self, batch_size: int, device="cpu") -> dict:
        """Muestrea un batch de secuencias contiguas de longitud seq_len.

        Args:
            batch_size: número de secuencias independientes en el batch (dimensión B).
            device: dispositivo PyTorch de los tensores resultantes.

        Returns:
            Dict de tensores [B, T, ...] en float32 (incluida `prev_action`, ya en one-hot).
            `cameras` y `bev` se guardan en uint8 y aquí se convierten a float32.
            `cameras` además se normaliza a [0, 1] dividiendo entre 255 (las demás claves ya
            estaban en su rango natural, así que solo cambian de tipo).

        Raises:
            RuntimeError: si no hay episodios con longitud >= seq_len.
        """
        valid_episodes = [episode for episode in self.episodes if len(episode) >= self.seq_len]
        if not valid_episodes:
            raise RuntimeError(
                f"sample() llamado sin episodios de longitud >= {self.seq_len}. "
                "Comprueba can_sample() antes de llamar a sample()."
            )

        keys = valid_episodes[0][0].keys()
        sequences = {k: [] for k in keys}

        # Muestrea batch_size ventanas aleatorias de longitud seq_len de episodios distintos.
        for _ in range(batch_size):
            episode = random.choice(valid_episodes)
            start_idx = random.randint(0, len(episode) - self.seq_len)
            step_sequence = episode[start_idx : start_idx + self.seq_len]
            for k in keys:
                sequences[k].append(np.stack([step[k] for step in step_sequence], 0))  # [T, ...]

        # Apila las B secuencias de cada clave en un tensor [B, T, ...] listo para el modelo.
        # cameras/bev viven en uint8 en memoria. Aquí se pasan a float32, y solo
        # cameras se normaliza (/255) porque bev ya es {0,1}.
        batch = {}
        for k in keys:
            stacked = np.stack(sequences[k], 0)     # [B, T, ...]
            tensor = torch.as_tensor(stacked, device=device).float()
            batch[k] = normalize_sensor_value(k, tensor)
        return batch
