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
DANGER_DISTANCE = CAR_LENGTH * 1.8
OPTIMAL_DISTANCE = CAR_LENGTH * 3.0
GUARDING_DISTANCE = SENSOR_RANGE  # Distancia de monitoreo (80 cm)

# Parámetros de los obstáculos (en cm)
NUM_RECTANGLES = (
    15  # Número de obstáculos rectangulares (reducido para espacio rectangular)
)
MIN_OBSTACLE_SIZE = 8.0  # Tamaño mínimo de los obstáculos en cm
MAX_OBSTACLE_SIZE = 25.0  # Tamaño máximo de los obstáculos en cm
MIN_DISTANCE_BETWEEN_OBSTACLE = (
    CAR_LENGTH * 2
)  # Distancia mínima entre obstáculos (30 cm)


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
        self.state["v_left"] = np.clip(
            self.state["v_left"], -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED
        )
        self.state["v_right"] = np.clip(
            self.state["v_right"], -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED
        )

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
        if not hasattr(self, "prev_pos"):
            self.prev_pos = np.array([self.state["x"], self.state["y"]])
        if not hasattr(self, "prev_speed"):
            self.prev_speed = 0.0

        current_pos = np.array([self.state["x"], self.state["y"]])

        # Para debugging
        if self.render_mode == "human":
            print(
                f"\nAcción ejecutada: {self.discrete_actions.describe_action(discrete_action)}"
            )

        # Obtener lecturas de sensores
        sensor_frontal = observation[1]
        sensor_izq = observation[3]
        sensor_der = observation[4]

        # Solo usar sensores relevantes (frontal, izquierdo, derecho)
        # El sensor trasero no es útil para el robot real
        min_sensor = np.min([sensor_frontal, sensor_izq, sensor_der])

        # ============================================================
        # SISTEMA DE RECOMPENSAS OPTIMIZADO - ROBUSTO Y ESCALABLE
        # ============================================================
        # Principios:
        # 1. Evitar colisiones (prioridad máxima)
        # 2. Mantener velocidad constante y positiva (objetivo principal)
        # 3. Exploración eficiente (cubrir área sin giros excesivos)
        # 4. Movimiento fluido (cambios suaves de velocidad)

        # Inicializar flags de terminación
        terminated = False
        truncated = False

        # ===== COMPONENTE 1: RECOMPENSA BASE POR NO COLISIONAR (Crucial) =====
        # Proporcional a la distancia mínima de los sensores
        # Normalizado: 0 en distancia de peligro, 1 en rango total del sensor
        min_distance_normalized = np.clip(min_sensor / SENSOR_RANGE, 0.0, 1.0)
        safety_reward = min_distance_normalized * 3.0  # Max 3.0 puntos
        reward += safety_reward

        # ===== COMPONENTE 2: PENALIZACIÓN POR PROXIMIDAD CRÍTICA =====
        # Penalización suave y gradual (no extrema) cuando se aproxima a obstáculos
        if min_sensor < DANGER_DISTANCE:
            # Penalización suave inversamente proporcional
            proximity_danger = 1.0 - (min_sensor / DANGER_DISTANCE)
            danger_penalty = -5.0 * proximity_danger  # Max -5.0 puntos
            reward += danger_penalty

        # ===== COMPONENTE 3: RECOMPENSA POR VELOCIDAD CONSTANTE (Objetivo Principal) =====
        # Incentiva mantener velocidad alta y consistente
        target_speed = MAX_WHEEL_SPEED * 0.7  # Velocidad objetivo: 70% del máximo
        speed_error = abs(self.state["speed"] - target_speed)

        # Penalización cuadrática por desviarse de velocidad objetivo
        # Esto incentiva velocidad cercana a la target
        speed_tracking_reward = -0.1 * (speed_error / MAX_WHEEL_SPEED) ** 2
        reward += speed_tracking_reward

        # Bonus por mantener velocidad positiva (hacia adelante)
        if (
            self.state["speed"] > MAX_WHEEL_SPEED * 0.5
        ):  # Si va al menos al 50% del máximo
            forward_bonus = 2.0
            reward += forward_bonus
        elif self.state["speed"] > 0:  # Si va hacia adelante pero lentamente
            forward_bonus = 0.5
            reward += forward_bonus
        else:  # Penalización por velocidad nula o negativa
            reward -= 1.5

        # ===== COMPONENTE 4: RECOMPENSA POR ESTABILIDAD DE VELOCIDAD (Suavidad) =====
        # Penaliza cambios abruptos de velocidad
        speed_change = abs(self.state["speed"] - self.prev_speed)
        smoothness_penalty = -0.2 * (speed_change / (ACCELERATION * dt))
        reward += smoothness_penalty

        # ===== COMPONENTE 5: INCENTIVO DE ACCIÓN INTELIGENTE =====
        # Refuerza acciones que mantienen velocidad alta sin cambios abruptos
        action_idx = discrete_action.action_idx

        if action_idx == 0:
            # ADELANTE puro: Máxima recompensa (acción ideal)
            reward += 2.0
        elif action_idx in [1, 3]:
            # Giros suaves (una rueda apagada): Recompensa moderada
            reward += 0.5
        elif action_idx == 4:
            # DETENIDO: Penalización moderada (detiene progreso)
            reward -= 1.0
        elif action_idx in [2, 6]:
            # Giros cerrados (una rueda atrás): Pequeña penalización
            reward -= 0.3
        elif action_idx in [5, 7]:
            # Acciones con una rueda atrás y otra apagada: Penalización leve
            reward -= 0.5
        elif action_idx == 8:
            # REVERSA completa: Penalización moderada (innecesaria en espacio abierto)
            reward -= 1.0

        # ===== COMPONENTE 6: EXPLORACIÓN EFICIENTE =====
        # Pequeña recompensa por exploración (cobertura de área)
        distance_traveled = np.linalg.norm(current_pos - self.prev_pos)
        if distance_traveled > 0.1:  # Solo si hay movimiento significativo
            exploration_bonus = 0.2 * min(distance_traveled / 5.0, 1.0)  # Max 0.2
            reward += exploration_bonus

        # ===== COMPONENTE 7: PENALIZACIÓN PROGRESIVA POR COLISIÓN =====
        # Escalable: penalización menor al inicio, mayor cuando ya debería saber evitar
        progress_ratio = self.step_count / self.max_steps

        # Terminar si hay colisión con pared u obstáculo
        if min_sensor < self.car_width:
            # Penalización escalable y razonable (no extrema)
            # Penaliza más al agente que aprendió (progreso > 20%)
            base_collision_penalty = -10.0
            if progress_ratio > 0.2:
                # Penalización aumentada después de episodios iniciales
                collision_penalty = base_collision_penalty * (1.0 + progress_ratio)
            else:
                # Penalización leve al inicio para permitir exploración
                collision_penalty = base_collision_penalty

            reward += collision_penalty

            if self.render_mode == "human":
                print(f"\n¡COLISIÓN! Penalización: {collision_penalty:.2f}")
                print(
                    f"Sobrevivió {self.step_count} pasos ({progress_ratio * 100:.1f}%)"
                )
            terminated = True

        # Terminar si sale del mapa
        elif not (
            0 <= self.state["x"] <= MAP_WIDTH and 0 <= self.state["y"] <= MAP_HEIGHT
        ):
            # Penalización salida del mapa (menos grave que colisión)
            boundary_penalty = -8.0 * (1.0 + progress_ratio * 0.5)
            reward += boundary_penalty

            if self.render_mode == "human":
                print(f"\n¡FUERA DEL MAPA! Penalización: {boundary_penalty:.2f}")
                print(
                    f"Sobrevivió {self.step_count} pasos ({progress_ratio * 100:.1f}%)"
                )
            terminated = True

        # ===== COMPONENTE 8: BONUS POR COMPLETAR EPISODIO (Éxito) =====
        elif self.step_count >= self.max_steps:
            # Bonus moderado por completar sin colisionar
            completion_bonus = 50.0
            reward += completion_bonus

            if self.render_mode == "human":
                print("\n¡EPISODIO COMPLETADO EXITOSAMENTE!")
                print(f"Bonus de completitud: +{completion_bonus:.2f}")
            truncated = True

        # Actualizar estado anterior
        self.prev_pos = current_pos.copy()
        self.prev_speed = self.state["speed"]

        # Incrementar el contador de pasos
        self.step_count += 1

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
            pygame.display.set_caption(
                f"Robot Simulator - {MAP_WIDTH:.0f}x{MAP_HEIGHT:.0f} cm"
            )
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
                    cm_to_pixels(height),
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
            (body_length / 2, -body_width / 2),  # Frente izquierda
            (body_length / 2, body_width / 2),  # Frente derecha
            (-body_length / 3, body_width / 2),  # Atrás derecha
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
            (
                -body_length / 3,
                -cm_to_pixels(WHEEL_BASE) / 2,
            ),  # Rueda trasera izquierda
            (-body_length / 3, cm_to_pixels(WHEEL_BASE) / 2),  # Rueda trasera derecha
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
        caster_rx = caster_x * cos(self.state["theta"]) - caster_y * sin(
            self.state["theta"]
        )
        caster_ry = caster_x * sin(self.state["theta"]) + caster_y * cos(
            self.state["theta"]
        )
        caster_pos = (int(car_pos[0] + caster_rx), int(car_pos[1] + caster_ry))

        # Dibujar rueda loca
        pygame.draw.circle(
            canvas, (100, 100, 100), caster_pos, int(caster_radius)
        )  # Gris

        # Dibujar marcador direccional (flecha roja en el frente)
        arrow_length = body_length * 0.3
        arrow_width = body_width * 0.3
        arrow_base_x = body_length / 2
        arrow_tip_x = body_length / 2 + arrow_length

        arrow_points = [
            (arrow_base_x, -arrow_width / 2),  # Base izquierda
            (arrow_tip_x, 0),  # Punta
            (arrow_base_x, arrow_width / 2),  # Base derecha
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
