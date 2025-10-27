# Robot Autónomo con Deep Reinforcement Learning

Este proyecto implementa un entorno de simulación 2D para el entrenamiento de un robot autónomo utilizando Deep Reinforcement Learning (DRL). El robot está equipado con sensores de distancia y debe aprender a navegar evitando obstáculos.

## Demo del Proyecto

https://github.com/user-attachments/assets/de5623ab-79f2-4ddb-9c0d-f0991bbacecb

## Características

-  Simulación 2D de un robot con dirección Ackermann
-  4 sensores de distancia (frontal, trasero, izquierdo, derecho)
-  Sistema de recompensas adaptativo
-  Implementación de DQN (Deep Q-Network)
-  Sistema de checkpointing y recuperación automática
-  Visualización en tiempo real

## Estructura del Proyecto

```
cu-robot-autonomo/
├── crt_car_env/              # Entorno de simulación
│   └── envs/
        ├── acions.py         # implementacion de las acciones que puede usar el carro
│       ├── car_env.py        # Implementación del entorno
├── rl_algoritmos/            # Algoritmos de RL
│   ├── agents/
│   │   └── dqn_agent.py      # Implementación del agente DQN
│   └── train_dqn.py         # Script de entrenamiento
└── checkpoints/             # Directorio para guardar modelos
```

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/apowopa/cu-robot-autonomo.git
cd cu-robot-autonomo
```

2. Crea un entorno virtual de python

3. Instalar el entorno:
```bash
pip install -e .
```

## Uso

### Entrenamiento Básico

```bash
python rl_algoritmos/train_dqn.py --tag mi_experimento --episodes 1000
```

### Opciones de Entrenamiento

- `--tag`: Identificador para los checkpoints (default: "default")
- `--episodes`: Número de episodios de entrenamiento (default: 1000)
- `--epsilon`: Valor inicial de epsilon para exploración (default: 1.0)
- `--render`: Activar visualización del entrenamiento
- `--debug`: Activar modo debug con información detallada
- `--checkpoint`: Ruta a un checkpoint para continuar el entrenamiento

### Ejemplos de Uso

1. Entrenamiento con visualización:
```bash
python rl_algoritmos/train_dqn.py --tag test --episodes 100 --render
```

2. Continuar desde un checkpoint:
```bash
python rl_algoritmos/train_dqn.py --tag test --checkpoint checkpoints/dqn_test_final.pth
```

3. Modo debug con visualización:
```bash
python rl_algoritmos/train_dqn.py --tag test --episodes 10 --debug --render
```

## Características del Entorno

### Estado
- Orientación del robot (-π a π)
- Lecturas de los 4 sensores de distancia

### Acciones
- ADELANTE
- REVERSA
- GIRO_IZQUIERDA
- GIRO_DERECHA

### Sistema de Recompensas
- Recompensas positivas por:
  * Avanzar en zonas seguras
  * Giros evasivos efectivos
  * Supervivencia prolongada

- Penalizaciones por:
  * Colisiones
  * Aproximación peligrosa a obstáculos
  * Comportamiento errático

### Sistema de Checkpointing
- Guardado automático cada 250 episodios
- Guardado del mejor modelo basado en rendimiento
- Sistema de fallback si el rendimiento empeora
- Recuperación automática ante interrupciones

## Configuración Avanzada

Los principales parámetros configurables se encuentran en:

1. `car_env.py`:
- `MAP_SIZE`: Tamaño del mapa
- `CAR_LENGTH`: Longitud del robot
- `SENSOR_RANGE`: Alcance de los sensores
- `NUM_RECTANGLES`: Número de obstáculos

2. `dqn_agent.py`:
- `hidden_size`: Tamaño de las capas ocultas
- `buffer_size`: Tamaño del buffer de experiencia
- `batch_size`: Tamaño del batch de entrenamiento
- `gamma`: Factor de descuento
- `tau`: Factor de actualización suave

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

