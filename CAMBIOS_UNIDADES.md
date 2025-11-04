# Cambios al Sistema de Unidades y Mapa

## Resumen de Cambios

Se ha modificado el entorno del robot para usar **centímetros (cm)** como unidad base en lugar de píxeles, y se ha cambiado de un mapa cuadrado a uno **rectangular**.

## 🗺️ Nuevo Mapa

### Dimensiones Anteriores
- Mapa: 800x800 píxeles (cuadrado)
- Sin conversión a unidades reales

### Dimensiones Nuevas
- **Ancho**: 300 cm (3 metros)
- **Alto**: 200 cm (2 metros)
- **Escala de visualización**: 2 píxeles por centímetro
- **Ventana de renderizado**: 600x400 píxeles

## 📏 Sistema de Unidades

### Antes (Píxeles)
```python
MAP_SIZE = 800.0  # píxeles
CAR_LENGTH = 40.0  # píxeles
WHEEL_BASE = 40.0  # píxeles
MAX_WHEEL_SPEED = 200.0  # píxeles/s
SENSOR_RANGE = 200.0  # píxeles
```

### Ahora (Centímetros)
```python
MAP_WIDTH = 300.0   # cm (3 metros)
MAP_HEIGHT = 200.0  # cm (2 metros)
CAR_LENGTH = 15.0   # cm
WHEEL_BASE = 12.0   # cm
MAX_WHEEL_SPEED = 50.0  # cm/s
SENSOR_RANGE = 80.0  # cm
PIXELS_PER_CM = 2.0  # Para renderizado
```

## 🚗 Dimensiones del Robot

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| Longitud | 15 cm | Largo total del robot |
| Ancho entre ruedas | 12 cm | Distancia entre ruedas motrices |
| Velocidad máxima | 50 cm/s | 0.5 m/s por rueda |

## 📡 Sensores

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| Rango máximo | 80 cm | Distancia máxima de detección |
| Distancia de peligro | 45 cm | Zona roja (3x largo del robot) |
| Distancia óptima | 67.5 cm | Distancia ideal para seguir paredes |

## 🎯 Obstáculos

| Parámetro | Antes | Ahora |
|-----------|-------|-------|
| Cantidad | 25 | 15 (ajustado para mapa rectangular) |
| Tamaño mínimo | 20 píxeles | 8 cm |
| Tamaño máximo | 75 píxeles | 25 cm |
| Distancia mínima | 80 píxeles | 30 cm |

## 🔄 Conversión Píxeles ↔ Centímetros

### En el Código Interno
Todas las operaciones físicas (cinemática, colisiones, sensores) usan **centímetros**.

### En el Renderizado
Se aplica una escala de **2 píxeles por centímetro**:
```python
def cm_to_pixels(value):
    return int(value * PIXELS_PER_CM)
```

## 🎨 Visualización

La ventana de pygame ahora tiene:
- **Ancho**: 600 píxeles (300 cm × 2)
- **Alto**: 400 píxeles (200 cm × 2)
- **Título**: Muestra las dimensiones reales

## 📊 Ventajas del Nuevo Sistema

1. **Unidades reales**: Más fácil de relacionar con el robot físico
2. **Escalabilidad**: Fácil ajustar el tamaño del mapa manteniendo proporciones
3. **Precisión**: Mejor correspondencia con medidas del mundo real
4. **Flexibilidad**: Mapa rectangular se adapta mejor a espacios reales
5. **Visualización**: Escala ajustable independiente de la física

## 🧪 Cómo Probar

```bash
# Probar el entorno con visualización
python test_rectangular_map.py

# Entrenar con el nuevo entorno
python drl_agents/train_dqn.py --tag nuevo_mapa --episodes 100 --render
```

## 📝 Archivos Modificados

1. **`crt_car_env/envs/car_env.py`**
   - Cambiadas todas las constantes a centímetros
   - Actualizado renderizado con conversión píxel/cm
   - Añadidas constantes de clase para acceso externo
   - Mapa rectangular en lugar de cuadrado

2. **`test_rectangular_map.py`** (nuevo)
   - Script de prueba del nuevo sistema
   - Muestra todas las medidas en cm
   - Verifica funcionamiento correcto

## ⚠️ Compatibilidad

Los modelos entrenados con el sistema anterior **no son directamente compatibles** porque:
- Las escalas de distancia han cambiado
- El espacio del mapa es diferente
- Las velocidades están en diferentes unidades

Se recomienda **reentrenar los modelos** con el nuevo sistema.

## 🔧 Personalización

Para ajustar el mapa, modifica estas constantes en `car_env.py`:

```python
# Tamaño del mapa (en cm)
MAP_WIDTH = 300.0   # Cambia el ancho
MAP_HEIGHT = 200.0  # Cambia el alto

# Escala de visualización
PIXELS_PER_CM = 2.0  # Más píxeles = mayor zoom
```

## 📐 Comparación de Escala

### Mapa Anterior
```
800×800 píxeles
Sin referencia real
Robot: 40×40 píxeles
```

### Mapa Actual
```
300×200 cm = 3×2 metros
600×400 píxeles (visualización)
Robot: 15×12 cm (real) → 30×24 píxeles (pantalla)
```

## 🎯 Casos de Uso

El nuevo sistema es ideal para:
- **Simulación realista**: Dimensiones coinciden con robot real
- **Pruebas de algoritmos**: Parámetros en unidades comprensibles
- **Validación**: Fácil verificar si comportamiento es físicamente posible
- **Transferencia**: Mejor transferencia de simulación a robot real

## 📞 Notas Adicionales

- Todas las velocidades ahora están en **cm/s**
- Los sensores reportan distancias en **cm**
- Las recompensas están ajustadas a la nueva escala
- El framerate sigue siendo 30 FPS
