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
    """Espacio de acciones discretas para el carro con tracción diferencial.
    
    Cada rueda puede estar en 3 estados:
    - ADELANTE: velocidad = +1.0 (encendida hacia adelante)
    - APAGADA: velocidad = 0.0 (motor apagado)
    - ATRÁS: velocidad = -1.0 (encendida hacia atrás)
    
    Esto da 9 combinaciones posibles (3 estados x 3 estados).
    """

    def __init__(self):
        """Inicializa el espacio de acciones con valores predefinidos.
        
        En el modelo de tracción diferencial:
        - speed: estado de la rueda izquierda (-1.0, 0.0, o 1.0)
        - steering: estado de la rueda derecha (-1.0, 0.0, o 1.0)
        """
        # Definir las 9 acciones posibles (combinaciones de estados de las ruedas)
        self.basic_actions = [
            # Rueda Izq: ADELANTE (+1), Rueda Der: ADELANTE (+1)
            Action(speed=1.0, steering=1.0, action_idx=0),
            
            # Rueda Izq: ADELANTE (+1), Rueda Der: APAGADA (0)
            Action(speed=1.0, steering=0.0, action_idx=1),
            
            # Rueda Izq: ADELANTE (+1), Rueda Der: ATRÁS (-1)
            Action(speed=1.0, steering=-1.0, action_idx=2),
            
            # Rueda Izq: APAGADA (0), Rueda Der: ADELANTE (+1)
            Action(speed=0.0, steering=1.0, action_idx=3),
            
            # Rueda Izq: APAGADA (0), Rueda Der: APAGADA (0)
            Action(speed=0.0, steering=0.0, action_idx=4),
            
            # Rueda Izq: APAGADA (0), Rueda Der: ATRÁS (-1)
            Action(speed=0.0, steering=-1.0, action_idx=5),
            
            # Rueda Izq: ATRÁS (-1), Rueda Der: ADELANTE (+1)
            Action(speed=-1.0, steering=1.0, action_idx=6),
            
            # Rueda Izq: ATRÁS (-1), Rueda Der: APAGADA (0)
            Action(speed=-1.0, steering=0.0, action_idx=7),
            
            # Rueda Izq: ATRÁS (-1), Rueda Der: ATRÁS (-1)
            Action(speed=-1.0, steering=-1.0, action_idx=8),
        ]

        self.n = len(self.basic_actions)
        self.all_actions = self.basic_actions

        # Mapeo de acciones para descripciones legibles
        self.action_descriptions = {
            0: "ADELANTE (IZQ:+, DER:+)",
            1: "GIRO_DERECHA_SUAVE (IZQ:+, DER:0)",
            2: "GIRO_DERECHA_CERRADO (IZQ:+, DER:-)",
            3: "GIRO_IZQUIERDA_SUAVE (IZQ:0, DER:+)",
            4: "DETENIDO (IZQ:0, DER:0)",
            5: "GIRO_IZQUIERDA_REVERSA (IZQ:0, DER:-)",
            6: "GIRO_IZQUIERDA_CERRADO (IZQ:-, DER:+)",
            7: "GIRO_DERECHA_REVERSA (IZQ:-, DER:0)",
            8: "REVERSA (IZQ:-, DER:-)",
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

