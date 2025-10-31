import numpy as np
from dataclasses import dataclass


@dataclass
class Action:
    """Clase para representar una acción del carro."""

    speed: float  # Velocidad normalizada (-1 a 1)
    steering: float  # Dirección normalizada (-1 a 1)
    action_idx: int  # Índice de la acción discreta

    def __str__(self) -> str:
        return f"Action(idx={self.action_idx}, speed={self.speed:.2f}, steering={self.steering:.2f})"

    def to_array(self) -> np.ndarray:
        """Convierte la acción a un array numpy para el ambiente."""
        return np.array([self.speed, self.steering], dtype=np.float32)


class DiscreteActionSpace:
    """Espacio de acciones discretas para el carro con tracción diferencial."""

    def __init__(self):
        """Inicializa el espacio de acciones con valores predefinidos.
        
        En el modelo de tracción diferencial:
        - speed: velocidad de la rueda izquierda (-1 a 1)
        - steering: velocidad de la rueda derecha (-1 a 1)
        """
        # Definir las acciones básicas
        self.basic_actions = [
            Action(speed=0.8, steering=0.8, action_idx=0),    # ADELANTE (ambas ruedas igual velocidad)
            Action(speed=-0.5, steering=-0.5, action_idx=1),  # REVERSA (ambas ruedas hacia atrás)
            Action(speed=0.3, steering=0.8, action_idx=2),    # GIRO IZQUIERDA (rueda izq lenta, der rápida)
            Action(speed=0.8, steering=0.3, action_idx=3),    # GIRO DERECHA (rueda izq rápida, der lenta)
            Action(speed=-0.3, steering=0.8, action_idx=4),   # GIRO CERRADO IZQUIERDA (rueda izq reversa, der adelante)
            Action(speed=0.8, steering=-0.3, action_idx=5),   # GIRO CERRADO DERECHA (rueda izq adelante, der reversa)
        ]

        self.n = len(self.basic_actions)
        self.all_actions = self.basic_actions

        # Mapeo de acciones para descripciones
        self.action_descriptions = {
            0: "ADELANTE",
            1: "REVERSA",
            2: "GIRO_IZQUIERDA",
            3: "GIRO_DERECHA",
            4: "GIRO_CERRADO_IZQUIERDA",
            5: "GIRO_CERRADO_DERECHA",
        }
 
    def sample(self) -> Action:
        """Selecciona una acción aleatoria."""
        idx = np.random.randint(0, self.n)
        return self.all_actions[idx]

    def get_action(self, action_idx: int) -> Action:
        """Obtiene una acción específica por su índice."""
        if not 0 <= action_idx < self.n:
            raise ValueError(f"action_idx debe estar entre 0 y {self.n - 1}")
        return self.all_actions[action_idx]

    def describe_action(self, action: Action) -> str:
        """
        Gener a una descripción legible de la acción.

        Args:
            action: Objeto Action a describir

        Returns:
            str: Descripción de la acción
        """
        return self.action_descriptions.get(action.action_idx, "DESCONOCIDO")

