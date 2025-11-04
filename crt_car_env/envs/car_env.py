import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
from math import sin, cos, pi
from .actions import DiscreteActionSpace

# Parámetros de la Simulación (en centímetros)
MAP_WIDTH = 200.0  # Ancho del mapa en cm 
MAP_HEIGHT = 350.0  # Alto del mapa en cm 
CAR_LENGTH = 15.0  # Longitud del coche en cm
WHEEL_BASE = 12.0  # Distancia entre las ruedas traseras (ancho del robot) en cm
MAX_WHEEL_SPEED = 50.0  # Velocidad máxima de cada rueda en cm/s
SENSOR_RANGE = 80.0  # Rango máximo de los sensores en cm

# Parámetros de física (inercia y fricción)
ACCELERATION = 150.0  # Aceleración máxima en cm/s² (cambio de velocidad)
FRICTION_COEFFICIENT = 0.15  # Coeficiente de fricción (0-1, mayor = más fricción)

# Parámetros de visualización
PIXELS_PER_CM = 3.0  # Factor de escala para el renderizado (2 píxeles por cm)

# Distancias críticas para los sensores (en función del largo del carro)
DANGER_DISTANCE = CAR_LENGTH * 1.5  
OPTIMAL_DISTANCE = CAR_LENGTH * 2.0  
GUARDING_DISTANCE = SENSOR_RANGE  # Distancia de monitoreo (80 cm)

# Parámetros de los obstáculos (en cm)
NUM_RECTANGLES = 15  # Número de obstáculos rectangulares (reducido para espacio rectangular)
MIN_OBSTACLE_SIZE = 8.0  # Tamaño mínimo de los obstáculos en cm
MAX_OBSTACLE_SIZE = 25.0  # Tamaño máximo de los obstáculos en cm
MIN_DISTANCE_BETWEEN_OBSTACLE = CAR_LENGTH * 2  # Distancia mínima entre obstáculos (30 cm)


class CRTCarEnv(gym.Env):
    """
    Entorno de simulación 2D para un robot con tracción diferencial
    (dos ruedas motrices traseras y una rueda loca frontal)
    y 4 sensores de distancia, compatible con Gymnasium.
    
    Todas las distancias están en centímetros (cm).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    # Constantes de clase (para acceso externo)
    MAP_WIDTH = MAP_WIDTH
    MAP_HEIGHT = MAP_HEIGHT
    CAR_LENGTH = CAR_LENGTH
    WHEEL_BASE = WHEEL_BASE
    MAX_WHEEL_SPEED = MAX_WHEEL_SPEED
    SENSOR_RANGE = SENSOR_RANGE
    PIXELS_PER_CM = PIXELS_PER_CM
    DANGER_DISTANCE = DANGER_DISTANCE
    OPTIMAL_DISTANCE = OPTIMAL_DISTANCE
    NUM_RECTANGLES = NUM_RECTANGLES
    MIN_OBSTACLE_SIZE = MIN_OBSTACLE_SIZE
    MAX_OBSTACLE_SIZE = MAX_OBSTACLE_SIZE

    def __init__(self, render_mode=None):
        super().__init__()

        # Espacio de observación:
        # [orientación, sensor_frontal, sensor_trasero, sensor_izquierdo, sensor_derecho]
        low_obs = np.array([-pi, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high_obs = np.array(
            [pi, SENSOR_RANGE, SENSOR_RANGE, SENSOR_RANGE, SENSOR_RANGE],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=low_obs, high=high_obs, dtype=np.float32
        )

        self.step_count = 0
        self.max_steps = 60 * self.metadata["render_fps"]

        # Configurar el espacio de acciones discreto
        self.discrete_actions = DiscreteActionSpace()
        self.action_space = spaces.Discrete(self.discrete_actions.n)

        # Espacio de acción interno (para el modelo físico de tracción diferencial):
        # [velocidad_rueda_izquierda, velocidad_rueda_derecha]
        # Ambos son valores continuos entre -1.0 y 1.0
        self._continuous_action_space = spaces.Box(
            low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32
        )

        # Estado interno del coche
        self.state = {
            "x": 0.0,  # Posición X (cm)
            "y": 0.0,  # Posición Y (cm)
            "theta": 0.0,  # Orientación (radianes)
            "speed": 0.0,  # Velocidad lineal actual (cm/s)
            "v_left": 0.0,  # Velocidad actual rueda izquierda (cm/s)
            "v_right": 0.0,  # Velocidad actual rueda derecha (cm/s)
        }

        # Variables de renderizado
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.car_width = CAR_LENGTH / 2.0

        # Lista de obstáculos
        self.obstacles = {
            "rectangles": [],  # Lista de [x, y, width, height]
        }

    def _get_sensor_readings(self):
        """
        Calcula las lecturas de los sensores mediante ray casting.
        Detecta tanto paredes como obstáculos.
        """
        readings = np.zeros(4)
        # [frontal, trasero, izquierdo, derecho]
        sensor_angles = [0, pi, pi / 6, -pi / 6]

        for i, angle in enumerate(sensor_angles):
            abs_angle = self.state["theta"] + angle
            direction = np.array([cos(abs_angle), sin(abs_angle)])

            # Ray casting
            current_pos = np.array([self.state["x"], self.state["y"]])
            for step in range(int(SENSOR_RANGE)):
                current_pos += direction

                # Verificar colisión con paredes
                if (
                    current_pos[0] < 0
                    or current_pos[0] >= MAP_WIDTH
                    or current_pos[1] < 0
                    or current_pos[1] >= MAP_HEIGHT
                ):
                    readings[i] = np.linalg.norm(
                        current_pos - np.array([self.state["x"], self.state["y"]])
                    )
                    break

                # Verificar colisión con obstáculos
                if self._check_obstacle_collision(current_pos):
                    readings[i] = step + 1  # +1 para evitar lecturas de 0
                    break
            else:
                readings[i] = SENSOR_RANGE

        return readings

    def _get_obs(self):
        """Genera la observación que el agente recibirá."""
        sensor_readings = self._get_sensor_readings()
        return np.array(
            [
                self.state["theta"],
                sensor_readings[0],  # Sensor Frontal
                sensor_readings[1],  # Sensor Trasero
                sensor_readings[2],  # Sensor Izquierdo
                sensor_readings[3],  # Sensor Derecho
            ],
            dtype=np.float32,
        )

    def _calculate_kinematics(self, action, dt):
        """
        Aplica el modelo cinemático de tracción diferencial con inercia.
        
        El robot tiene dos ruedas motrices traseras independientes y una rueda loca frontal.
        Las velocidades de las ruedas cambian gradualmente (aceleración/desaceleración)
        en lugar de instantáneamente, simulando la inercia del sistema.

        Args:
            action (np.ndarray): [Velocidad objetivo rueda izquierda (-1 a 1), 
                                  Velocidad objetivo rueda derecha (-1 a 1)]
            dt (float): Paso de tiempo en segundos.
        """
        left_wheel_input, right_wheel_input = action

        # Velocidades objetivo (deseadas) para cada rueda
        v_left_target = left_wheel_input * MAX_WHEEL_SPEED
        v_right_target = right_wheel_input * MAX_WHEEL_SPEED

        # Aplicar aceleración gradual con inercia
        # La velocidad cambia hacia el objetivo, limitada por la aceleración máxima
        
        # Calcular la diferencia entre velocidad actual y objetivo
        delta_v_left = v_left_target - self.state["v_left"]
        delta_v_right = v_right_target - self.state["v_right"]
        
        # Limitar el cambio de velocidad según la aceleración máxima
        max_delta_v = ACCELERATION * dt
        
        # Aplicar aceleración limitada
        if abs(delta_v_left) > max_delta_v:
            delta_v_left = max_delta_v * np.sign(delta_v_left)
        if abs(delta_v_right) > max_delta_v:
            delta_v_right = max_delta_v * np.sign(delta_v_right)
        
        # Actualizar velocidades actuales
        self.state["v_left"] += delta_v_left
        self.state["v_right"] += delta_v_right
        
        # Aplicar fricción (desaceleración cuando no hay input)
        # La fricción reduce la velocidad proporcionalmente
        friction_factor = 1.0 - FRICTION_COEFFICIENT * dt
        if abs(left_wheel_input) < 0.01:  # Si no hay input significativo
            self.state["v_left"] *= friction_factor
        if abs(right_wheel_input) < 0.01:
            self.state["v_right"] *= friction_factor
        
        # Limitar velocidades al máximo permitido
        self.state["v_left"] = np.clip(self.state["v_left"], -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
        self.state["v_right"] = np.clip(self.state["v_right"], -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)

        # Calcular velocidad lineal del centro del robot (promedio de las ruedas)
        v = (self.state["v_left"] + self.state["v_right"]) / 2.0
        
        # Calcular velocidad angular (diferencia de velocidades / distancia entre ruedas)
        omega = (self.state["v_right"] - self.state["v_left"]) / WHEEL_BASE

        # Actualizar orientación
        self.state["theta"] += omega * dt
        # Normalizar a [-pi, pi]
        self.state["theta"] = (self.state["theta"] + pi) % (2 * pi) - pi

        # Actualizar posición del centro del robot
        self.state["x"] += v * cos(self.state["theta"]) * dt
        self.state["y"] += v * sin(self.state["theta"]) * dt
        
        # Guardar la velocidad lineal actual para referencia
        self.state["speed"] = v

    def step(self, action):
        """
        Ejecuta un paso de tiempo en el entorno.
        """
        dt = 1.0 / self.metadata["render_fps"]  # Paso de tiempo basado en FPS

        # Convertir acción discreta a continua
        discrete_action = self.discrete_actions.get_action(action)
        continuous_action = discrete_action.to_array()

        # Aplicar acción y actualizar estado
        self._calculate_kinematics(continuous_action, dt)

        # Obtener nueva observación
        observation = self._get_obs()

        # Calcular recompensa
        reward = 0.0
        
        # Guardar estado anterior para calcular progreso
        if not hasattr(self, 'prev_pos'):
            self.prev_pos = np.array([self.state["x"], self.state["y"]])
        
        current_pos = np.array([self.state["x"], self.state["y"]])

        # Para debugging
        if self.render_mode == "human":
            print(
                f"\nAcción ejecutada: {self.discrete_actions.describe_action(discrete_action)}"
            )

        # Obtener lecturas de sensores
        sensor_frontal = observation[1]
        sensor_trasero = observation[2]  # No se usa en recompensas
        sensor_izq = observation[3]
        sensor_der = observation[4]
        
        # Solo usar sensores relevantes (frontal, izquierdo, derecho)
        # El sensor trasero no es útil para el robot real
        min_sensor = np.min([sensor_frontal, sensor_izq, sensor_der])

        # ============================================================
        # SISTEMA DE RECOMPENSAS
        # ============================================================
        
        # 1. RECOMPENSA POR MOVIMIENTO HACIA ADELANTE (muy incentivado)
        distance_traveled = np.linalg.norm(current_pos - self.prev_pos)
        
        # Calcular si el movimiento fue hacia adelante (en dirección de orientación)
        if distance_traveled > 0.1:  # Solo si hubo movimiento significativo
            movement_vector = current_pos - self.prev_pos
            forward_direction = np.array([np.cos(self.state["theta"]), 
                                         np.sin(self.state["theta"])])
            
            # Producto punto para determinar si va hacia adelante
            forward_component = np.dot(movement_vector, forward_direction)
            
            if forward_component > 0:
                # Movimiento hacia adelante: GRAN recompensa
                movement_reward = forward_component * 2.0  # Duplicado
                reward += movement_reward
            else:
                # Movimiento hacia atrás: penalización
                reward -= abs(forward_component) * 0.5
        
        # 2. RECOMPENSA POR VELOCIDAD POSITIVA (ir hacia adelante)
        if self.state["speed"] > 0:
            # Recompensa proporcional a velocidad hacia adelante
            speed_reward = self.state["speed"] * 0.3
            reward += speed_reward
        else:
            # Penalización por velocidad negativa o cero
            reward -= 1.0
        
        # 3. RECOMPENSA/PENALIZACIÓN FUERTE POR TIPO DE ACCIÓN
        action_desc = self.discrete_actions.describe_action(discrete_action)
        
        if discrete_action == 0:
            # ADELANTE puro (ambas ruedas +): MÁXIMA recompensa
            reward += 5.0  # Aumentado significativamente
        elif discrete_action in [1, 3]:
            # Giros suaves manteniendo adelante: recompensa moderada
            reward += 1.0
        elif discrete_action == 2:
            # Giro cerrado derecha (una rueda reversa): penalización moderada
            reward -= 2.0
        elif discrete_action == 4:
            # DETENIDO: penalización fuerte
            reward -= 5.0  # Aumentado
        elif discrete_action in [5, 6, 7]:
            # Acciones con reversa: penalización muy fuerte
            reward -= 8.0  # Aumentado
        elif discrete_action == 8:
            # REVERSA completa: penalización extrema
            reward -= 10.0  # Aumentado
        
        # 5. RECOMPENSA POR MANTENER DISTANCIA SEGURA (evitar obstáculos)
        # Solo usar sensores frontal, izquierdo y derecho (el trasero no es útil)
        front_safety = sensor_frontal / SENSOR_RANGE
        left_safety = sensor_izq / SENSOR_RANGE
        right_safety = sensor_der / SENSOR_RANGE
        
        # Recompensa por espacio libre (solo sensores relevantes)
        avg_clearance = (front_safety + left_safety + right_safety) / 3.0
        clearance_reward = avg_clearance * 1.0
        reward += clearance_reward
        
        # 5. PENALIZACIÓN POR PROXIMIDAD PELIGROSA
        if min_sensor < DANGER_DISTANCE:
            danger_ratio = min_sensor / DANGER_DISTANCE
            danger_penalty = -5.0 * (1.0 - danger_ratio) ** 2  # Penalización cuadrática
            reward += danger_penalty
        
        # 6. RECOMPENSA POR EXPLORACIÓN (incentivar cubrir área)
        # Pequeña recompensa por cada paso sin colisión
        exploration_reward = 0.3
        reward += exploration_reward
        
        # 7. PENALIZACIÓN POR USO EXCESIVO DE GIROS CERRADOS
        # (incentivar movimiento suave)
        if "CERRADO" in action_desc:
            reward -= 0.5
        
        # 8. BONUS POR MOVIMIENTO FLUIDO
        # Recompensa si la velocidad es similar al paso anterior
        if hasattr(self, 'prev_speed'):
            speed_change = abs(self.state["speed"] - self.prev_speed)
            if speed_change < 5.0:  # Cambio suave de velocidad
                fluidity_bonus = 0.3
                reward += fluidity_bonus
        
        # Actualizar estado anterior
        self.prev_pos = current_pos
        self.prev_speed = self.state["speed"]
        
        # Incrementar el contador de pasos
        self.step_count += 1
        
        # 9. RECOMPENSA POR SUPERVIVENCIA (cada paso que sobrevive)
        # Recompensa incremental que aumenta con el tiempo
        survival_progress = self.step_count / self.max_steps
        survival_reward = 0.5 + (survival_progress * 2.0)  # De 0.5 a 2.5
        reward += survival_reward

        # Verificar terminación
        terminated = False
        truncated = False

        # Terminar si hay colisión con pared u obstáculo
        if min_sensor < self.car_width:
            # PENALIZACIÓN MUY FUERTE inversamente proporcional al progreso
            # Cuanto más temprano muere, mayor es la penalización
            progress = self.step_count / self.max_steps
            
            # Penalización inversamente proporcional: 1/(progress + 0.01) para evitar división por cero
            # Escalar para que sea muy fuerte al principio
            base_penalty = 500.0  # Penalización base muy alta
            early_death_penalty = base_penalty / (progress + 0.02)  # +0.02 para suavizar un poco
            
            # Cap máximo de penalización para evitar valores extremos
            early_death_penalty = min(early_death_penalty, 2000.0)

            # Penalización extra si la colisión fue frontal con acción adelante
            if sensor_frontal == min_sensor and discrete_action == 0:
                early_death_penalty *= 1.5  # 50% extra por colisión frontal con adelante

            reward -= early_death_penalty
            if self.render_mode == "human":
                print(f"\n¡COLISIÓN! Penalización: -{early_death_penalty:.2f}")
                print(f"Sobrevivió {self.step_count} pasos ({progress*100:.1f}% del episodio)")
                if sensor_frontal == min_sensor and discrete_action == 0:
                    print("¡Colisión frontal con acción ADELANTE!")
            terminated = True

        # Terminar si sale del mapa
        if not (0 <= self.state["x"] <= MAP_WIDTH and 0 <= self.state["y"] <= MAP_HEIGHT):
            # Penalización inversamente proporcional similar
            progress = self.step_count / self.max_steps
            base_penalty = 400.0  # Ligeramente menor que colisión
            early_death_penalty = base_penalty / (progress + 0.02)
            early_death_penalty = min(early_death_penalty, 1500.0)

            reward -= early_death_penalty
            if self.render_mode == "human":
                print(f"\n¡FUERA DEL MAPA! Penalización: -{early_death_penalty:.2f}")
                print(f"Sobrevivió {self.step_count} pasos ({progress*100:.1f}% del episodio)")
            terminated = True

        # Terminar si se alcanza el máximo de pasos (ÉXITO COMPLETO)
        if self.step_count >= self.max_steps:
            # GRAN BONUS por completar el episodio
            completion_bonus = 100.0
            reward += completion_bonus
            if self.render_mode == "human":
                print("\n¡EPISODIO COMPLETADO! ¡Supervivencia máxima alcanzada!")
                print(f"Bonus de completitud: +{completion_bonus:.2f}")
            truncated = True
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def _generate_random_obstacles(self):
        """Genera obstáculos aleatorios en el mapa manteniendo una distancia mínima entre ellos."""
        self.obstacles["rectangles"] = []
        self.obstacles["circles"] = []

        margin = MIN_DISTANCE_BETWEEN_OBSTACLE  # Margen para evitar obstáculos muy cerca de los bordes
        max_attempts = 100  # Número máximo de intentos por obstáculo

        # Generar rectángulos aleatorios
        for _ in range(NUM_RECTANGLES):
            placed = False
            for attempt in range(max_attempts):
                width = self.np_random.uniform(MIN_OBSTACLE_SIZE, MAX_OBSTACLE_SIZE)
                height = self.np_random.uniform(MIN_OBSTACLE_SIZE, MAX_OBSTACLE_SIZE)
                x = self.np_random.uniform(margin, MAP_WIDTH - width - margin)
                y = self.np_random.uniform(margin, MAP_HEIGHT - height - margin)

                # Verificar distancia con otros obstáculos
                valid_position = True
                for other_obs in self.obstacles["rectangles"]:
                    ox, oy, ow, oh = other_obs
                    # Calcular centros
                    center1 = np.array([x + width / 2, y + height / 2])
                    center2 = np.array([ox + ow / 2, oy + oh / 2])
                    # Calcular distancia entre centros
                    distance = np.linalg.norm(center1 - center2)
                    if distance < MIN_DISTANCE_BETWEEN_OBSTACLE:
                        valid_position = False
                        break

                if valid_position:
                    self.obstacles["rectangles"].append([x, y, width, height])
                    placed = True
                    break

            if not placed and self.render_mode == "human":
                print(
                    f"Advertencia: No se pudo colocar el obstáculo {_ + 1} después de {max_attempts} intentos"
                )

    def _check_obstacle_collision(self, position):
        """
        Verifica si hay colisión con algún obstáculo.

        Args:
            position: np.array([x, y]) - Posición a verificar

        Returns:
            bool: True si hay colisión, False en caso contrario
        """
        x, y = position

        # Verificar colisión con rectángulos
        for rx, ry, width, height in self.obstacles["rectangles"]:
            if rx <= x <= rx + width and ry <= y <= ry + height:
                return True

        # Verificar colisión con círculos
        for cx, cy, radius in self.obstacles["circles"]:
            if np.linalg.norm(np.array([x - cx, y - cy])) <= radius:
                return True

        return False

    def _is_valid_position(self, position):
        """
        Verifica si una posición es válida (sin colisiones y dentro del mapa).
        """
        x, y = position
        if not (0 <= x <= MAP_WIDTH and 0 <= y <= MAP_HEIGHT):
            return False

        # Verificar colisión con obstáculos
        if self._check_obstacle_collision(position):
            return False

        return True

    def reset(self, *, seed=None, options=None):
        """
        Reinicia el entorno con un nuevo seed.

        Args:
            seed: Semilla para el generador de números aleatorios
            options: Opciones adicionales (no utilizadas actualmente)

        Returns:
            observation: Primera observación del episodio
            info: Información adicional del estado
        """
        # Inicializar el generador de números aleatorios
        super().reset(seed=seed)

        # Asegurarse de que numpy.random también use la misma semilla
        if seed is not None:
            np.random.seed(seed)

        # Reiniciar el contador de pasos
        self.step_count = 0

        # Generar nuevo mapa de obstáculos
        self._generate_random_obstacles()

        # Encontrar una posición inicial válida para el coche
        margin = CAR_LENGTH * 2
        max_attempts = 100

        for _ in range(max_attempts):
            x = self.np_random.uniform(margin, MAP_WIDTH - margin)
            y = self.np_random.uniform(margin, MAP_HEIGHT - margin)

            # Verificar si la posición es válida
            if self._is_valid_position(np.array([x, y])):
                self.state["x"] = x
                self.state["y"] = y
                self.state["theta"] = self.np_random.uniform(-pi, pi)
                self.state["speed"] = 0.0
                self.state["v_left"] = 0.0
                self.state["v_right"] = 0.0
                break
        else:
            # Si no se encuentra una posición válida, usar el centro del mapa
            if self.render_mode == "human":
                print("Advertencia: No se encontró una posición inicial válida")
            self.state["x"] = MAP_WIDTH / 2
            self.state["y"] = MAP_HEIGHT / 2
            self.state["theta"] = 0.0
            self.state["speed"] = 0.0
            self.state["v_left"] = 0.0
            self.state["v_right"] = 0.0

        # Inicializar variables para el sistema de recompensas
        self.prev_pos = np.array([self.state["x"], self.state["y"]])
        self.prev_speed = 0.0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _get_info(self):
        """Información adicional sobre el estado."""
        return {
            "x_position": self.state["x"],
            "y_position": self.state["y"],
            "speed": self.state["speed"],
            "orientation": self.state["theta"],
        }

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        # Calcular dimensiones de la ventana en píxeles
        window_width = int(MAP_WIDTH * PIXELS_PER_CM)
        window_height = int(MAP_HEIGHT * PIXELS_PER_CM)
        
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((window_width, window_height))
            pygame.display.set_caption(f"Robot Simulator - {MAP_WIDTH:.0f}x{MAP_HEIGHT:.0f} cm")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((window_width, window_height))
        canvas.fill((255, 255, 255))  # Fondo blanco
        
        # Función auxiliar para convertir coordenadas de cm a píxeles
        def cm_to_pixels(value):
            return int(value * PIXELS_PER_CM)

        # Dibujar obstáculos
        # Rectángulos
        for x, y, width, height in self.obstacles["rectangles"]:
            pygame.draw.rect(
                canvas,
                (128, 128, 128),  # Gris
                pygame.Rect(
                    cm_to_pixels(x),
                    cm_to_pixels(y),
                    cm_to_pixels(width),
                    cm_to_pixels(height)
                ),
            )

        # Círculos
        for x, y, radius in self.obstacles["circles"]:
            pygame.draw.circle(
                canvas,
                (100, 100, 100),  # Gris oscuro
                (cm_to_pixels(x), cm_to_pixels(y)),
                cm_to_pixels(radius),
            )

        # Dibujar el robot con tracción diferencial
        car_pos = (cm_to_pixels(self.state["x"]), cm_to_pixels(self.state["y"]))

        # Dimensiones del robot (en píxeles para el renderizado)
        body_width = cm_to_pixels(WHEEL_BASE * 0.8)  # Ancho del cuerpo
        body_length = cm_to_pixels(CAR_LENGTH)  # Largo del cuerpo
        wheel_width = cm_to_pixels(WHEEL_BASE * 0.15)  # Ancho de las ruedas
        wheel_length = cm_to_pixels(CAR_LENGTH * 0.25)  # Largo de las ruedas
        caster_radius = cm_to_pixels(WHEEL_BASE * 0.12)  # Radio de la rueda loca

        # Puntos del cuerpo principal (rectangular)
        body_points = [
            (-body_length / 3, -body_width / 2),  # Atrás izquierda
            (body_length / 2, -body_width / 2),   # Frente izquierda
            (body_length / 2, body_width / 2),    # Frente derecha
            (-body_length / 3, body_width / 2),   # Atrás derecha
        ]

        # Rotar y dibujar el cuerpo principal
        rotated_body = []
        for x, y in body_points:
            rx = x * cos(self.state["theta"]) - y * sin(self.state["theta"])
            ry = x * sin(self.state["theta"]) + y * cos(self.state["theta"])
            rotated_body.append((int(car_pos[0] + rx), int(car_pos[1] + ry)))

        # Dibujar cuerpo principal del robot
        pygame.draw.polygon(canvas, (0, 0, 255), rotated_body)  # Azul

        # Dibujar las dos ruedas motrices traseras
        rear_wheel_positions = [
            (-body_length / 3, -cm_to_pixels(WHEEL_BASE) / 2),  # Rueda trasera izquierda
            (-body_length / 3, cm_to_pixels(WHEEL_BASE) / 2),   # Rueda trasera derecha
        ]

        for wx, wy in rear_wheel_positions:
            wheel_points = [
                (wx - wheel_length / 2, wy - wheel_width / 2),
                (wx + wheel_length / 2, wy - wheel_width / 2),
                (wx + wheel_length / 2, wy + wheel_width / 2),
                (wx - wheel_length / 2, wy + wheel_width / 2),
            ]

            rotated_wheel = []
            for x, y in wheel_points:
                rx = x * cos(self.state["theta"]) - y * sin(self.state["theta"])
                ry = x * sin(self.state["theta"]) + y * cos(self.state["theta"])
                rotated_wheel.append((int(car_pos[0] + rx), int(car_pos[1] + ry)))
            pygame.draw.polygon(canvas, (0, 0, 0), rotated_wheel)  # Ruedas negras

        # Dibujar la rueda loca frontal (pequeño círculo)
        caster_x = body_length / 2
        caster_y = 0
        
        # Rotar posición de la rueda loca
        caster_rx = caster_x * cos(self.state["theta"]) - caster_y * sin(self.state["theta"])
        caster_ry = caster_x * sin(self.state["theta"]) + caster_y * cos(self.state["theta"])
        caster_pos = (int(car_pos[0] + caster_rx), int(car_pos[1] + caster_ry))
        
        # Dibujar rueda loca
        pygame.draw.circle(canvas, (100, 100, 100), caster_pos, int(caster_radius))  # Gris

        # Dibujar marcador direccional (flecha roja en el frente)
        arrow_length = body_length * 0.3
        arrow_width = body_width * 0.3
        arrow_base_x = body_length / 2
        arrow_tip_x = body_length / 2 + arrow_length
        
        arrow_points = [
            (arrow_base_x, -arrow_width / 2),   # Base izquierda
            (arrow_tip_x, 0),                   # Punta
            (arrow_base_x, arrow_width / 2),    # Base derecha
        ]
        
        rotated_arrow = []
        for x, y in arrow_points:
            rx = x * cos(self.state["theta"]) - y * sin(self.state["theta"])
            ry = x * sin(self.state["theta"]) + y * cos(self.state["theta"])
            rotated_arrow.append((int(car_pos[0] + rx), int(car_pos[1] + ry)))
        
        # Dibujar flecha direccional
        pygame.draw.polygon(canvas, (255, 0, 0), rotated_arrow)  # Rojo

        # Dibujar sensores
        sensor_readings = self._get_sensor_readings()
        sensor_angles = [
            0,
            pi,
            pi / 6,
            -pi / 6,
        ]  # [frontal, trasero, izquierdo, derecho]

        for reading, angle in zip(sensor_readings, sensor_angles):
            abs_angle = self.state["theta"] + angle

            # Punto donde termina la zona roja (umbral de guardia) - convertir a píxeles
            guard_x = car_pos[0] + cos(abs_angle) * cm_to_pixels(DANGER_DISTANCE)
            guard_y = car_pos[1] + sin(abs_angle) * cm_to_pixels(DANGER_DISTANCE)

            # Punto final del sensor (donde detecta algo) - convertir a píxeles
            end_x = car_pos[0] + cos(abs_angle) * cm_to_pixels(reading)
            end_y = car_pos[1] + sin(abs_angle) * cm_to_pixels(reading)

            # Dibujar parte roja (desde el carro hasta el umbral)
            pygame.draw.line(
                canvas, (255, 0, 0), car_pos, (int(guard_x), int(guard_y)), 2
            )

            # Dibujar parte verde (desde el umbral hasta donde detecta algo)
            pygame.draw.line(
                canvas,
                (0, 255, 0),
                (int(guard_x), int(guard_y)),
                (int(end_x), int(end_y)),
                2,
            )

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
