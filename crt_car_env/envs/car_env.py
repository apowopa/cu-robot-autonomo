import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
from math import sin, cos, pi, tan
from .actions import DiscreteActionSpace, Action

# Parámetros de la Simulación
MAP_SIZE = 800.0  # Tamaño del mapa en píxeles
CAR_LENGTH = 40.0  # Longitud del coche (distancia entre ejes)
MAX_SPEED = 200.0  # Velocidad máxima en píxeles por segundo
MAX_STEER_ANGLE = pi / 4  # 45 grados
SENSOR_RANGE = 200.0  # Rango máximo de los sensores

# Distancias críticas para los sensores (en función del largo del carro)
DANGER_DISTANCE = CAR_LENGTH * 3.0     # Zona de peligro inmediato
OPTIMAL_DISTANCE = CAR_LENGTH * 4.5     # Distancia óptima para seguir paredes
GUARDING_DISTANCE = SENSOR_RANGE    # Distancia de monitoreo

# Parámetros de los obstáculos
NUM_RECTANGLES = 15  # Número de obstáculos rectangulares
MIN_OBSTACLE_SIZE = 30.0  # Tamaño mínimo de los obstáculos
MAX_OBSTACLE_SIZE = 75.0  # Tamaño máximo de los obstáculos
MIN_DISTANCE_BETWEEN_OBSTACLE = 80

class CRTCarEnv(gym.Env):
    """
    Entorno de simulación 2D para un coche RC con dirección Ackermann 
    y 4 sensores de distancia, compatible con Gymnasium.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        
        # Espacio de observación:
        # [orientación, sensor_frontal, sensor_trasero, sensor_izquierdo, sensor_derecho]
        low_obs = np.array([-pi, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high_obs = np.array([pi, SENSOR_RANGE, SENSOR_RANGE, SENSOR_RANGE, SENSOR_RANGE], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
        
        # Contador de pasos para el episodio actual
        self.step_count = 0
        # Máximo número de pasos por episodio (30 segundos a 30 FPS)
        self.max_steps = 60 * self.metadata["render_fps"]

        # Configurar el espacio de acciones discreto
        self.discrete_actions = DiscreteActionSpace()
        self.action_space = spaces.Discrete(self.discrete_actions.n)

        # Espacio de acción interno (para el modelo físico):
        # [velocidad, ángulo_dirección]
        # Ambos son valores continuos entre -1.0 y 1.0
        self._continuous_action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # Estado interno del coche
        self.state = {
            'x': 0.0,           # Posición X
            'y': 0.0,           # Posición Y
            'theta': 0.0,       # Orientación (radianes)
            'speed': 0.0        # Velocidad lineal actual
        }

        # Variables de renderizado
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.car_width = CAR_LENGTH / 2.0
        
        # Lista de obstáculos
        self.obstacles = {
            'rectangles': [],  # Lista de [x, y, width, height]
        }

    def _get_sensor_readings(self):
        """
        Calcula las lecturas de los sensores mediante ray casting.
        Detecta tanto paredes como obstáculos.
        """
        readings = np.zeros(4)
        #[frontal, trasero, izquierdo, derecho]
        sensor_angles = [0, pi, pi/6, -pi/6]
        
        for i, angle in enumerate(sensor_angles):
            abs_angle = self.state['theta'] + angle
            direction = np.array([cos(abs_angle), sin(abs_angle)])
            
            # Ray casting
            current_pos = np.array([self.state['x'], self.state['y']])
            for step in range(int(SENSOR_RANGE)):
                current_pos += direction
                
                # Verificar colisión con paredes
                if (current_pos[0] < 0 or current_pos[0] >= MAP_SIZE or
                    current_pos[1] < 0 or current_pos[1] >= MAP_SIZE):
                    readings[i] = np.linalg.norm(
                        current_pos - np.array([self.state['x'], self.state['y']])
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
        return np.array([
            self.state['theta'],
            sensor_readings[0],  # Sensor Frontal
            sensor_readings[1],  # Sensor Trasero
            sensor_readings[2],  # Sensor Izquierdo
            sensor_readings[3]   # Sensor Derecho
        ], dtype=np.float32)

    def _calculate_kinematics(self, action, dt):
        """
        Aplica el modelo cinemático de Ackermann para actualizar la posición.
        
        Args:
            action (np.ndarray): [Velocidad (-1 a 1), Ángulo de Dirección (-1 a 1)]
            dt (float): Paso de tiempo en segundos.
        """
        speed_input, steer_input = action
        
        # Mapear entradas a valores físicos
        desired_speed = speed_input * MAX_SPEED
        # Aplicar un modelo simple de aceleración/desaceleración
        speed_diff = desired_speed - self.state['speed']
        self.state['speed'] += speed_diff * dt * 2.0  # Factor 2.0 para respuesta más rápida
        
        # Mapear el ángulo de dirección
        steer_angle = steer_input * MAX_STEER_ANGLE
        
        # Modelo cinemático de Ackermann
        if abs(self.state['speed']) > 0.1:  # Si hay movimiento significativo
            # Velocidad angular
            angular_velocity = (self.state['speed'] / CAR_LENGTH) * tan(steer_angle)
            
            # Actualizar orientación
            self.state['theta'] += angular_velocity * dt
            # Normalizar a [-pi, pi]
            self.state['theta'] = (self.state['theta'] + pi) % (2 * pi) - pi
            
            # Actualizar posición
            self.state['x'] += self.state['speed'] * cos(self.state['theta']) * dt
            self.state['y'] += self.state['speed'] * sin(self.state['theta']) * dt

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
        
        # Para debugging
        if self.render_mode == "human":
            print(f"\nAcción ejecutada: {self.discrete_actions.describe_action(discrete_action)}")

        # Obtener lecturas de sensores
        sensor_frontal = observation[1]  # Lectura del sensor frontal
        sensor_izq = observation[3]      # Lectura del sensor izquierdo
        sensor_der = observation[4]      # Lectura del sensor derecho
        min_sensor = np.min([sensor_frontal, sensor_izq, sensor_der])
        
        # 1. Recompensas base por acción
        action_desc = self.discrete_actions.describe_action(discrete_action)
        
        # Penalización gradual por proximidad frontal
        if sensor_frontal < OPTIMAL_DISTANCE:
            # Calcular qué tan cerca está del obstáculo
            proximity_factor = (OPTIMAL_DISTANCE - sensor_frontal) / (OPTIMAL_DISTANCE - DANGER_DISTANCE)
            proximity_factor = np.clip(proximity_factor, 0, 1)  # Asegurar que esté entre 0 y 1
            
            # Penalización exponencial por proximidad
            base_penalty = -5.0
            proximity_penalty = base_penalty * (proximity_factor ** 2)  # Penalización cuadrática
            
            # Detectar comportamiento suicida (acercamiento frontal)
            is_frontal_approach = (sensor_frontal < OPTIMAL_DISTANCE and 
                                sensor_izq > OPTIMAL_DISTANCE and 
                                sensor_der > OPTIMAL_DISTANCE)
            
            # Detectar si se está moviendo hacia el obstáculo
            if action_desc == "ADELANTE" and is_frontal_approach:
                proximity_penalty *= 3.0  # Triple penalización por avanzar directamente hacia un obstáculo
                if self.render_mode == "human":
                    print("\n¡Alerta! Comportamiento suicida detectado")
            
            reward += proximity_penalty
        
        # Recompensas por acciones
        if action_desc == "ADELANTE":
            # Zona segura
            if sensor_frontal > OPTIMAL_DISTANCE:
                safe_space = min(sensor_izq, sensor_der) / SENSOR_RANGE
                base_reward = 2.0
                space_bonus = safe_space * 2.0  # Bonus por espacio libre a los lados
                reward += base_reward + space_bonus
            else:
                # Penalización progresiva por avanzar hacia el peligro
                danger_factor = (OPTIMAL_DISTANCE - sensor_frontal) / OPTIMAL_DISTANCE
                penalty = -15.0 * danger_factor  # Penalización más severa
                reward += penalty
                
        elif action_desc == "REVERSA":
            if sensor_frontal < OPTIMAL_DISTANCE:
                # Recompensa por alejarse del peligro
                danger_factor = (OPTIMAL_DISTANCE - sensor_frontal) / OPTIMAL_DISTANCE
                reversa_reward = 4.0 * danger_factor
                
                # Bonus extra por evitar colisión inminente
                if sensor_frontal < DANGER_DISTANCE:
                    reversa_reward *= 1.5
                    
                reward += reversa_reward
            else:
                # Penalización moderada por reversa innecesaria
                # Menor penalización si hay obstáculos cercanos a los lados
                if min(sensor_izq, sensor_der) < OPTIMAL_DISTANCE:
                    reward -= 1.0
                else:
                    reward -= 2.0
                
        elif action_desc in ["GIRO_IZQUIERDA", "GIRO_DERECHA"]:
            # Detectar si es necesario girar
            need_to_turn = sensor_frontal < OPTIMAL_DISTANCE * 1.5
            
            if need_to_turn:
                # Determinar la dirección óptima de giro
                space_left = sensor_izq
                space_right = sensor_der
                turning_left = action_desc == "GIRO_IZQUIERDA"
                
                # Calcular el diferencial de espacio normalizado
                space_diff = abs(space_left - space_right) / SENSOR_RANGE
                
                if turning_left:
                    if space_left > space_right:
                        # Giro correcto hacia el espacio más amplio
                        base_reward = 5.0
                        space_bonus = space_diff * 5.0
                        urgency_bonus = (OPTIMAL_DISTANCE - sensor_frontal) / OPTIMAL_DISTANCE
                        reward += base_reward + space_bonus + (urgency_bonus * 3.0)
                    else:
                        # Giro hacia el espacio más reducido
                        penalty_factor = space_diff + 0.5  # Penalización base + diferencia
                        reward -= 4.0 * penalty_factor
                else:  # Girando a la derecha
                    if space_right > space_left:
                        # Giro correcto hacia el espacio más amplio
                        base_reward = 5.0
                        space_bonus = space_diff * 5.0
                        urgency_bonus = (OPTIMAL_DISTANCE - sensor_frontal) / OPTIMAL_DISTANCE
                        reward += base_reward + space_bonus + (urgency_bonus * 3.0)
                    else:
                        # Giro hacia el espacio más reducido
                        penalty_factor = space_diff + 0.5
                        reward -= 4.0 * penalty_factor
            else:
                # Giro innecesario - penalización variable según la proximidad de obstáculos
                closest_obstacle = min(sensor_frontal, sensor_izq, sensor_der)
                safety_factor = closest_obstacle / OPTIMAL_DISTANCE
                safety_factor = np.clip(safety_factor, 0, 1)
                
                # Penalización más suave si hay obstáculos cercanos
                penalty = -2.0 * safety_factor
                reward += penalty
        
        # Penalizaciones por situaciones extremadamente peligrosas
        if min_sensor < DANGER_DISTANCE:
            # Penalización exponencial por proximidad al peligro
            danger_factor = (DANGER_DISTANCE - min_sensor) / DANGER_DISTANCE
            danger_penalty = -5.0 * (danger_factor ** 2)  # Penalización cuadrática
            reward += danger_penalty
        
        # Incrementar el contador de pasos
        self.step_count += 1
        
        # Recompensa por supervivencia (curva logarítmica)
        # La recompensa crece más rápido al principio y se estabiliza con el tiempo
        normalized_time = self.step_count / self.max_steps
        survival_reward = 0.5 * np.log(1 + 10 * normalized_time)  # Base 0.5, factor 10 para ajustar la curva
        reward += survival_reward
        
        # Verificar terminación
        terminated = False
        truncated = False
        
        # Terminar si hay colisión con pared u obstáculo
        if min_sensor < self.car_width:
            # Penalización exponencial basada en el tiempo de supervivencia
            # La penalización es máxima al principio y disminuye exponencialmente
            progress = self.step_count / self.max_steps
            base_penalty = 150.0  # Penalización base aumentada
            early_death_penalty = base_penalty * np.exp(-3 * progress)  # Factor -3 para ajustar la curva
            
            # Penalización extra si la colisión fue frontal
            if sensor_frontal == min_sensor and action_desc == "ADELANTE":
                early_death_penalty *= 1.5  # 50% extra por colisión frontal intencional
            
            reward -= early_death_penalty
            if self.render_mode == "human":
                print(f"\nColisión detectada! Penalización: {early_death_penalty:.2f}")
                print(f"Tiempo de supervivencia: {self.step_count} pasos")
                if sensor_frontal == min_sensor and action_desc == "ADELANTE":
                    print("¡Colisión frontal intencional detectada!")
            terminated = True
        
        # Terminar si sale del mapa
        if not (0 <= self.state['x'] <= MAP_SIZE and 0 <= self.state['y'] <= MAP_SIZE):
            # Penalización similar a la colisión pero ligeramente menor
            progress = self.step_count / self.max_steps
            base_penalty = 120.0  # Menor que la penalización por colisión
            early_death_penalty = base_penalty * np.exp(-3 * progress)
            
            reward -= early_death_penalty
            if self.render_mode == "human":
                print(f"\nSalió del mapa! Penalización: {early_death_penalty:.2f}")
                print(f"Tiempo de supervivencia: {self.step_count} pasos")
            terminated = True
            
        # Terminar si se alcanza el máximo de pasos
        if self.step_count >= self.max_steps:
            if self.render_mode == "human":
                print("\n¡Episodio completado! Supervivencia máxima alcanzada.")
            truncated = True
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def _generate_random_obstacles(self):
        """Genera obstáculos aleatorios en el mapa manteniendo una distancia mínima entre ellos."""
        self.obstacles['rectangles'] = []
        self.obstacles['circles'] = []
        
        margin = CAR_LENGTH * 3  # Margen para evitar obstáculos muy cerca de los bordes
        max_attempts = 100  # Número máximo de intentos por obstáculo
        
        # Generar rectángulos aleatorios
        for _ in range(NUM_RECTANGLES):
            placed = False
            for attempt in range(max_attempts):
                width = self.np_random.uniform(MIN_OBSTACLE_SIZE, MAX_OBSTACLE_SIZE)
                height = self.np_random.uniform(MIN_OBSTACLE_SIZE, MAX_OBSTACLE_SIZE)
                x = self.np_random.uniform(margin, MAP_SIZE - width - margin)
                y = self.np_random.uniform(margin, MAP_SIZE - height - margin)
                
                # Verificar distancia con otros obstáculos
                valid_position = True
                for other_obs in self.obstacles['rectangles']:
                    ox, oy, ow, oh = other_obs
                    # Calcular centros
                    center1 = np.array([x + width/2, y + height/2])
                    center2 = np.array([ox + ow/2, oy + oh/2])
                    # Calcular distancia entre centros
                    distance = np.linalg.norm(center1 - center2)
                    if distance < MIN_DISTANCE_BETWEEN_OBSTACLE:
                        valid_position = False
                        break
                
                if valid_position:
                    self.obstacles['rectangles'].append([x, y, width, height])
                    placed = True
                    break
                    
            if not placed and self.render_mode == "human":
                print(f"Advertencia: No se pudo colocar el obstáculo {_+1} después de {max_attempts} intentos")

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
        for rx, ry, width, height in self.obstacles['rectangles']:
            if (rx <= x <= rx + width and 
                ry <= y <= ry + height):
                return True
        
        # Verificar colisión con círculos
        for cx, cy, radius in self.obstacles['circles']:
            if np.linalg.norm(np.array([x - cx, y - cy])) <= radius:
                return True
        
        return False

    def _is_valid_position(self, position):
        """
        Verifica si una posición es válida (sin colisiones y dentro del mapa).
        """
        x, y = position
        if not (0 <= x <= MAP_SIZE and 0 <= y <= MAP_SIZE):
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
            x = self.np_random.uniform(margin, MAP_SIZE - margin)
            y = self.np_random.uniform(margin, MAP_SIZE - margin)
            
            # Verificar si la posición es válida
            if self._is_valid_position(np.array([x, y])):
                self.state['x'] = x
                self.state['y'] = y
                self.state['theta'] = self.np_random.uniform(-pi, pi)
                self.state['speed'] = 0.0
                break
        else:
            # Si no se encuentra una posición válida, usar el centro del mapa
            if self.render_mode == "human":
                print("Advertencia: No se encontró una posición inicial válida")
            self.state['x'] = MAP_SIZE / 2
            self.state['y'] = MAP_SIZE / 2
            self.state['theta'] = 0.0
            self.state['speed'] = 0.0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _get_info(self):
        """Información adicional sobre el estado."""
        return {
            'x_position': self.state['x'],
            'y_position': self.state['y'],
            'speed': self.state['speed'],
            'orientation': self.state['theta']
        }

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((int(MAP_SIZE), int(MAP_SIZE)))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((int(MAP_SIZE), int(MAP_SIZE)))
        canvas.fill((255, 255, 255))  # Fondo blanco

        # Dibujar obstáculos
        # Rectángulos
        for x, y, width, height in self.obstacles['rectangles']:
            pygame.draw.rect(canvas, (128, 128, 128), # Gris
                           pygame.Rect(int(x), int(y), int(width), int(height)))
        
        # Círculos
        for x, y, radius in self.obstacles['circles']:
            pygame.draw.circle(canvas, (100, 100, 100), # Gris oscuro
                             (int(x), int(y)), int(radius))

        # Dibujar el coche
        car_pos = (int(self.state['x']), int(self.state['y']))
        
        # Dimensiones del carro
        car_width = self.car_width * 0.7  # Ancho del cuerpo
        wheel_width = car_width * 0.4     # Ancho de las ruedas
        wheel_length = CAR_LENGTH * 0.2   # Largo de las ruedas
        
        # Puntos del coche (forma rectangular)
        car_points = [
            (-CAR_LENGTH/2, -car_width),    # Atrás izquierda
            (CAR_LENGTH/2, -car_width),     # Frente izquierda
            (CAR_LENGTH/2, car_width),      # Frente derecha
            (-CAR_LENGTH/2, car_width),     # Atrás derecha
        ]
        
        # Rotar y dibujar el cuerpo principal
        rotated_points = []
        for x, y in car_points:
            rx = x * cos(self.state['theta']) - y * sin(self.state['theta'])
            ry = x * sin(self.state['theta']) + y * cos(self.state['theta'])
            rotated_points.append((
                int(car_pos[0] + rx),
                int(car_pos[1] + ry)
            ))
        
        # Dibujar cuerpo principal del coche
        pygame.draw.polygon(canvas, (0, 0, 255), rotated_points)  # Azul
        
        # Definir posiciones de las ruedas
        wheel_positions = [
            (-CAR_LENGTH/3, -car_width-wheel_width/2),  # Izquierda trasera
            (CAR_LENGTH/3, -car_width-wheel_width/2),   # Izquierda delantera
            (-CAR_LENGTH/3, car_width+wheel_width/2),   # Derecha trasera
            (CAR_LENGTH/3, car_width+wheel_width/2),    # Derecha delantera
        ]
        
        # Dibujar las ruedas
        for wx, wy in wheel_positions:
            wheel_points = [
                (wx - wheel_length/2, wy - wheel_width/2),
                (wx + wheel_length/2, wy - wheel_width/2),
                (wx + wheel_length/2, wy + wheel_width/2),
                (wx - wheel_length/2, wy + wheel_width/2),
            ]
            
            rotated_wheel = []
            for x, y in wheel_points:
                rx = x * cos(self.state['theta']) - y * sin(self.state['theta'])
                ry = x * sin(self.state['theta']) + y * cos(self.state['theta'])
                rotated_wheel.append((
                    int(car_pos[0] + rx),
                    int(car_pos[1] + ry)
                ))
            pygame.draw.polygon(canvas, (0, 0, 0), rotated_wheel)  # Ruedas negras
        
        # Dibujar la cabeza semicircular (roja)
        head_radius = car_width * 1.1  # Radio de la cabeza
        head_center_x = CAR_LENGTH/2
        head_center_y = 0
        
        # Rotar el centro de la cabeza
        head_rx = head_center_x * cos(self.state['theta']) - head_center_y * sin(self.state['theta'])
        head_ry = head_center_x * sin(self.state['theta']) + head_center_y * cos(self.state['theta'])
        head_center = (int(car_pos[0] + head_rx), int(car_pos[1] + head_ry))
        
        # Calcular los puntos para el semicírculo
        num_points = 12  # Número de puntos para aproximar el semicírculo
        head_points = [(head_center_x, 0)]  # Punto central
        
        for i in range(num_points + 1):
            angle = -pi/2 + (i * pi) / num_points  # De -90 a 90 grados
            x = head_center_x + head_radius * cos(angle)
            y = head_radius * sin(angle)
            head_points.append((x, y))
        
        # Rotar y convertir los puntos de la cabeza
        rotated_head = []
        for x, y in head_points:
            rx = x * cos(self.state['theta']) - y * sin(self.state['theta'])
            ry = x * sin(self.state['theta']) + y * cos(self.state['theta'])
            rotated_head.append((
                int(car_pos[0] + rx),
                int(car_pos[1] + ry)
            ))
        
        # Dibujar la cabeza semicircular
        pygame.draw.polygon(canvas, (255, 0, 0), rotated_head)  # Rojo
        
        # Dibujar sensores
        sensor_readings = self._get_sensor_readings()
        sensor_angles = [0, pi, pi/6, -pi/6]  # [frontal, trasero, izquierdo, derecho]
        
        for reading, angle in zip(sensor_readings, sensor_angles):
            abs_angle = self.state['theta'] + angle
            
            # Punto donde termina la zona roja (umbral de guardia)
            guard_x = car_pos[0] + cos(abs_angle) * DANGER_DISTANCE
            guard_y = car_pos[1] + sin(abs_angle) * DANGER_DISTANCE
            
            # Punto final del sensor (donde detecta algo)
            end_x = car_pos[0] + cos(abs_angle) * reading
            end_y = car_pos[1] + sin(abs_angle) * reading
            
            # Dibujar parte roja (desde el carro hasta el umbral)
            pygame.draw.line(canvas, (255, 0, 0), car_pos, (int(guard_x), int(guard_y)), 2)
            
            # Dibujar parte verde (desde el umbral hasta donde detecta algo)
            pygame.draw.line(canvas, (0, 255, 0), (int(guard_x), int(guard_y)), 
                           (int(end_x), int(end_y)), 2)

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
