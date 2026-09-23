"""DiscreteActionSpace: mapea acciones discretas para el control de CARLA (throttle, steer, brake).

Adaptación del espacio de 39 acciones de Raw2Drive: 13 ángulos de giro x 3 modos
(acelerar / rodar / frenar) = 39 acciones.
"""

from dataclasses import dataclass

DEFAULT_STEERS = [-0.6, -0.4, -0.25, -0.15, -0.08, -0.03, 0.0,
                  0.03, 0.08, 0.15, 0.25, 0.4, 0.6]  # 13 ángulos

@dataclass(frozen=True)
class Control:
    """Control de vehículo simplificado (subconjunto de carla.VehicleControl)."""
    throttle: float     # Aceleración
    steer: float        # Dirección
    brake: float        # Freno


class DiscreteActionSpace:
    """Acciones discretas -> (throttle, steer, brake). `n` = nº de acciones."""

    def __init__(self, steers=None, throttle: float = 0.6, brake: float = 0.6):
        """Construye la tabla de acciones a partir de los ángulos de giro y valores fijos.

        Args:
            steers: lista de ángulos de steer, por defecto DEFAULT_STEERS (13).
            throttle: valor fijo de aceleración usado en las acciones "acelerar".
            brake: valor fijo de frenado usado en las acciones "frenar".
        """
        self.steers = list(steers) if steers is not None else list(DEFAULT_STEERS)
        self.throttle = throttle
        self.brake = brake
        self._table = self._build()

    def _build(self):
        """Genera la tabla de Control: bloques acelerar / rodar / frenar, uno por ángulo de steer."""
        table = []
        for s in self.steers:                 # acelerar
            table.append(Control(self.throttle, s, 0.0))
        for s in self.steers:                 # rodar
            table.append(Control(0.0, s, 0.0))
        for s in self.steers:                 # frenar
            table.append(Control(0.0, s, self.brake))
        return table

    @property
    def n(self) -> int:
        """Número total de acciones discretas disponibles."""
        return len(self._table)

    def to_control(self, action: int) -> Control:
        """Traduce un índice de acción a su Control correspondiente.

        Raises:
            IndexError: si `action` está fuera de [0, n).
        """
        if not 0 <= action < self.n:
            raise IndexError(f"acción {action} fuera de rango [0, {self.n})")
        return self._table[action]

    def to_carla(self, action: int):
        """Devuelve carla.VehicleControl."""
        import carla
        control = self.to_control(action)

        return carla.VehicleControl(throttle=float(control.throttle),
                                    steer=float(control.steer),
                                    brake=float(control.brake))
