# Controlador del Robot con Agente DQN

Este módulo integra el agente DQN entrenado con el hardware del robot de tracción diferencial.

## Descripción

El controlador conecta:
- **Sensores VL53L0X**: 3 sensores de distancia (frontal, izquierda, derecha)
- **Motores de tracción diferencial**: 2 ruedas motrices traseras controladas independientemente
- **Agente DQN**: Red neuronal entrenada para la navegación autónoma

## Requisitos de Hardware

- Raspberry Pi (u otro dispositivo compatible con GPIO)
- 2 motores DC para tracción diferencial
- 3 sensores VL53L0X de distancia
- Driver de motores (L298N o similar)

### Conexiones GPIO

**Motores:**
- Motor Izquierdo A: GPIO 5
- Motor Izquierdo B: GPIO 6
- Motor Derecho A: GPIO 13
- Motor Derecho B: GPIO 19

**Sensores VL53L0X (XSHUT pins):**
- Sensor Frontal: GPIO 4
- Sensor Derecha: GPIO 17
- Sensor Izquierda: GPIO 27

## Instalación

```bash
# Instalar dependencias de hardware
pip install gpiozero lgpio adafruit-circuitpython-vl53l0x

# Instalar dependencias del proyecto
pip install -r requirements.txt
```

## Uso

### Ejecución básica

```bash
python agent_controller.py --checkpoint ../checkpoints/dqn_gigi.pth
```

### Opciones

```bash
python agent_controller.py \
  --checkpoint ../checkpoints/dqn_gigi.pth \
  --obstacle-distance 200 \
  --duration 60
```

**Parámetros:**
- `--checkpoint`: (requerido) Ruta al modelo entrenado (.pth)
- `--obstacle-distance`: Distancia mínima de obstáculo en mm (default: 200)
- `--duration`: Duración de ejecución en segundos (default: indefinido)

## Modelo de Tracción Diferencial

El robot usa un modelo de tracción diferencial con las siguientes características:

### Acciones disponibles

| Acción | Rueda Izq | Rueda Der | Descripción |
|--------|-----------|-----------|-------------|
| 0 | 0.8 | 0.8 | Adelante |
| 1 | -0.5 | -0.5 | Reversa |
| 2 | 0.3 | 0.8 | Giro Izquierda |
| 3 | 0.8 | 0.3 | Giro Derecha |
| 4 | -0.3 | 0.8 | Giro Cerrado Izquierda |
| 5 | 0.8 | -0.3 | Giro Cerrado Derecha |

### Cinemática

```
velocidad_lineal = (v_izq + v_der) / 2
velocidad_angular = (v_der - v_izq) / ancho_eje
```

Donde:
- `v_izq`: velocidad de la rueda izquierda
- `v_der`: velocidad de la rueda derecha
- `ancho_eje`: distancia entre las ruedas (0.1m)

## Estado del Agente

El agente recibe un vector de estado de 12 dimensiones:

1. **Distancia frontal** (normalizada 0-1)
2. **Distancia izquierda** (normalizada 0-1)
3. **Distancia derecha** (normalizada 0-1)
4. **Posición X** (normalizada)
5. **Posición Y** (normalizada)
6. **Orientación θ** (radianes)
7. **Objetivo X** (normalizada)
8. **Objetivo Y** (normalizada)
9. **Velocidad lineal**
10-12. **Información adicional**

## Estructura del Código

```python
class RobotController:
    - _setup_hardware(): Inicializa GPIO y sensores
    - _load_agent(): Carga el modelo DQN
    - read_sensors(): Lee distancias de los sensores
    - get_state(): Construye el estado para el agente
    - execute_action(): Ejecuta una acción en el hardware
    - run(): Loop principal del controlador
    - cleanup(): Limpia recursos al terminar
```

## Ejemplo de Uso Programático

```python
from agent_controller import RobotController

# Crear controlador
controller = RobotController(
    checkpoint_path='../checkpoints/dqn_gigi.pth',
    obstacle_distance=200
)

# Ejecutar por 60 segundos
controller.run(duration=60)
```

## Detener el Robot

Para detener el robot durante la ejecución:
- Presiona `Ctrl+C` en el terminal
- El sistema ejecutará automáticamente `cleanup()` para detener los motores

## Troubleshooting

### Error: "No se puede acceder a GPIO"
- Asegúrate de ejecutar con privilegios suficientes
- En Raspberry Pi: `sudo python agent_controller.py ...`

### Error: "No se puede inicializar sensor"
- Verifica las conexiones I2C
- Ejecuta `i2cdetect -y 1` para ver dispositivos conectados
- Revisa las conexiones de alimentación de los sensores

### El robot no se mueve
- Verifica las conexiones del driver de motores
- Comprueba la alimentación de los motores
- Revisa que los GPIOs estén correctamente configurados

## Comparación con Controlador Original

| Aspecto | Controlador Original | Agent Controller |
|---------|---------------------|------------------|
| Control | Reglas fijas (if/else) | Red neuronal (DQN) |
| Decisiones | Basadas en umbrales | Aprendidas |
| Adaptabilidad | Limitada | Alta |
| Física | Ackermann | Tracción diferencial |
| Sensores | 3 VL53L0X | 3 VL53L0X |

## Modelos Disponibles

En el directorio `checkpoints/`:
- `dqn_gigi.pth`: Modelo general
- `dqn_gura.pth`: Modelo específico
- `dqn_sora.pth`: Modelo experimental
- `dqn_ellie_interrupt_*.pth`: Checkpoints intermedios

## Próximos Pasos

1. **Integrar localización**: Usar IMU o encoders para posición real
2. **Agregar PWM**: Control de velocidad variable en lugar de on/off
3. **Sistema de waypoints**: Navegación a objetivos específicos
4. **Telemetría**: Enviar datos a servidor para monitoreo
5. **SLAM**: Construcción de mapa del entorno

## Licencia

Ver archivo LICENSE en la raíz del proyecto.
