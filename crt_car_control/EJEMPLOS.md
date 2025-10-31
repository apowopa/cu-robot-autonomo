# Ejemplos de Ejecución del Controlador del Robot

Este documento muestra diferentes formas de ejecutar el controlador del robot con el agente DQN.

## 📋 Prerrequisitos

Antes de ejecutar, asegúrate de:
1. Tener las dependencias instaladas: `pip install -r requirements.txt`
2. Tener al menos un modelo entrenado en `../checkpoints/`
3. Estar conectado al hardware del robot (Raspberry Pi)

## 🚀 Ejemplos de Línea de Comandos

### 1. Ejecución Básica (Modo Interactivo)

```bash
# Navegar al directorio del controlador
cd crt_car_control

# Ejecutar con el menú interactivo
python ejemplo_ejecucion.py
```

**Resultado:** Se mostrará un menú para seleccionar diferentes ejemplos.

---

### 2. Ejecución Directa con Modelo

```bash
# Ejecutar con el modelo por defecto (indefinido, presiona Ctrl+C para detener)
python agent_controller.py --checkpoint ../checkpoints/dqn_gigi.pth
```

**Resultado:** El robot inicia navegación autónoma hasta que presiones `Ctrl+C`.

---

### 3. Ejecución con Duración Limitada

```bash
# Ejecutar por 60 segundos
python agent_controller.py \
  --checkpoint ../checkpoints/dqn_gigi.pth \
  --duration 60
```

**Resultado:** El robot se ejecuta por exactamente 60 segundos y luego se detiene automáticamente.

---

### 4. Cambiar Distancia de Obstáculo

```bash
# Usar distancia de obstáculo de 30cm (más conservador)
python agent_controller.py \
  --checkpoint ../checkpoints/dqn_gigi.pth \
  --obstacle-distance 300
```

**Resultado:** El robot mantiene al menos 30cm de distancia a los obstáculos.

---

### 5. Ejecutar con Diferentes Modelos

```bash
# Modelo Gigi (general)
python agent_controller.py --checkpoint ../checkpoints/dqn_gigi.pth

# Modelo Gura (específico)
python agent_controller.py --checkpoint ../checkpoints/dqn_gura.pth

# Modelo Sora (experimental)
python agent_controller.py --checkpoint ../checkpoints/dqn_sora.pth

# Checkpoint intermedio de Ellie
python agent_controller.py --checkpoint ../checkpoints/dqn_ellie_interrupt_20251029_1743.pth
```

---

### 6. Ejecución Completa con Todas las Opciones

```bash
python agent_controller.py \
  --checkpoint ../checkpoints/dqn_gura.pth \
  --obstacle-distance 250 \
  --duration 120
```

**Configuración:**
- Modelo: `dqn_gura.pth`
- Distancia mínima: 25cm
- Duración: 2 minutos

---

### 7. Ejecutar como Servicio (Background)

```bash
# Ejecutar en background con nohup
nohup python agent_controller.py \
  --checkpoint ../checkpoints/dqn_gigi.pth \
  --duration 300 \
  > robot_log.txt 2>&1 &

# Ver el proceso
ps aux | grep agent_controller

# Ver logs en tiempo real
tail -f robot_log.txt

# Detener el proceso
kill $(pgrep -f agent_controller)
```

---

### 8. Ejecutar con Logging Detallado

```bash
# Redirigir output a archivo con timestamp
python agent_controller.py \
  --checkpoint ../checkpoints/dqn_gigi.pth \
  2>&1 | tee "robot_$(date +%Y%m%d_%H%M%S).log"
```

**Resultado:** Guarda todo el output en un archivo con timestamp.

---

### 9. Test de Hardware (Solo Sensores)

```bash
# Ejecutar el ejemplo de test de sensores
python -c "from ejemplo_ejecucion import ejemplo_test_sensores; ejemplo_test_sensores()"
```

**Resultado:** Lee los sensores sin mover el robot (útil para debugging).

---

### 10. Ejecutar con Privilegios (Raspberry Pi)

```bash
# En Raspberry Pi puede requerir permisos de root para GPIO
sudo python agent_controller.py --checkpoint ../checkpoints/dqn_gigi.pth
```

---

## 🐍 Ejemplos de Código Python

### Ejemplo Básico

```python
from agent_controller import RobotController

# Crear controlador
controller = RobotController(
    checkpoint_path='../checkpoints/dqn_gigi.pth',
    obstacle_distance=200
)

# Ejecutar
controller.run()
```

### Ejemplo con Try-Except

```python
from agent_controller import RobotController

try:
    controller = RobotController(
        checkpoint_path='../checkpoints/dqn_gigi.pth',
        obstacle_distance=200
    )
    controller.run(duration=60.0)
except KeyboardInterrupt:
    print("Robot detenido por el usuario")
except Exception as e:
    print(f"Error: {e}")
```

### Ejemplo con Monitoreo

```python
from agent_controller import RobotController
import time

controller = RobotController(
    checkpoint_path='../checkpoints/dqn_gigi.pth',
    obstacle_distance=200
)

# Monitorear sensores manualmente
for i in range(100):
    distances = controller.read_sensors()
    state = controller.get_state()
    
    print(f"Iteración {i}: {distances}")
    time.sleep(0.2)

controller.cleanup()
```

### Ejemplo con Control Manual

```python
from agent_controller import RobotController

controller = RobotController(
    checkpoint_path='../checkpoints/dqn_gigi.pth',
    obstacle_distance=200
)

try:
    # Ejecutar acción específica
    controller.execute_action(0)  # Adelante
    time.sleep(2)
    
    controller.execute_action(2)  # Giro izquierda
    time.sleep(1)
    
    controller.stop()
finally:
    controller.cleanup()
```

---

## 📊 Interpretación de la Salida

Durante la ejecución, verás algo como:

```
Configurando hardware...
Inicializando sensores...
Sensor 'izquierda' inicializado en 0x30
Sensor 'frontal' inicializado en 0x31
Sensor 'derecha' inicializado en 0x32
Hardware configurado correctamente.
Cargando agente desde ../checkpoints/dqn_gigi.pth...
Agente cargado correctamente.
Iniciando controlador del robot...
Ejecutando: ADELANTE | Rueda Izq=0.80, Der=0.80
Sensores: F= 450mm | I= 320mm | D= 380mm
Ejecutando: GIRO_DERECHA | Rueda Izq=0.80, Der=0.30
Sensores: F= 180mm | I= 280mm | D= 410mm
Ejecutando: ADELANTE | Rueda Izq=0.80, Der=0.80
Sensores: F= 520mm | I= 350mm | D= 380mm
...
```

**Leyenda:**
- `F`: Sensor frontal (mm)
- `I`: Sensor izquierdo (mm)
- `D`: Sensor derecho (mm)
- `Rueda Izq/Der`: Velocidad normalizada (-1 a 1)

---

## 🔧 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'gpiozero'"

```bash
# Instalar dependencias de hardware
pip install gpiozero lgpio adafruit-circuitpython-vl53l0x
```

### Error: "FileNotFoundError: checkpoint not found"

```bash
# Verificar que el checkpoint existe
ls -lh ../checkpoints/

# Usar ruta absoluta si es necesario
python agent_controller.py --checkpoint /home/apowo/Projects/cu-robot-autonomo/checkpoints/dqn_gigi.pth
```

### Error: "Permission denied" en GPIO

```bash
# Ejecutar con sudo en Raspberry Pi
sudo python agent_controller.py --checkpoint ../checkpoints/dqn_gigi.pth

# O agregar usuario al grupo gpio
sudo usermod -a -G gpio $USER
# Luego logout/login
```

### Robot no responde

```bash
# Verificar conexiones de hardware
i2cdetect -y 1  # Ver sensores I2C

# Test de sensores sin movimiento
python -c "from ejemplo_ejecucion import ejemplo_test_sensores; ejemplo_test_sensores()"
```

---

## 📝 Notas Importantes

1. **Seguridad**: Siempre supervisa el robot durante las pruebas iniciales
2. **Espacio**: Asegúrate de tener espacio libre para que el robot se mueva
3. **Batería**: Verifica que la batería esté cargada antes de iniciar
4. **Interrupciones**: Presiona `Ctrl+C` para detener de forma segura en cualquier momento
5. **Logs**: Guarda los logs para análisis posterior del comportamiento

---

## 🎯 Casos de Uso Comunes

### Desarrollo y Testing

```bash
# Test corto de 30 segundos
python agent_controller.py --checkpoint ../checkpoints/dqn_gigi.pth --duration 30
```

### Demostración

```bash
# Ejecución continua con distancia conservadora
python agent_controller.py --checkpoint ../checkpoints/dqn_gigi.pth --obstacle-distance 300
```

### Experimentos

```bash
# Probar diferentes modelos con configuración fija
for model in ../checkpoints/*.pth; do
    echo "Testing $model"
    python agent_controller.py --checkpoint "$model" --duration 60
    sleep 5
done
```

### Producción

```bash
# Ejecución robusta con logging
python agent_controller.py \
  --checkpoint ../checkpoints/dqn_gigi.pth \
  --obstacle-distance 200 \
  2>&1 | tee -a production_$(date +%Y%m%d).log
```

---

## 🔗 Referencias

- Ver `README.md` para más detalles sobre el sistema
- Ver `agent_controller.py` para documentación del código
- Ver `../drl_agents/train_dqn.py` para entrenamiento de modelos
