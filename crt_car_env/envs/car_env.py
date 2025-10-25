import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
from math import sin, cos, pi, tan

# Parámetros de la Simulación
MAP_SIZE = 800.0  # Tamaño del mapa en píxeles
CAR_LENGTH = 40.0  # Longitud del coche (distancia entre ejes)
MAX_SPEED = 200.0  # Velocidad máxima en píxeles por segundo
MAX_STEER_ANGLE = pi / 4  # 45 grados
SENSOR_RANGE = 200.0  # Rango máximo de los sensores

# Parámetros de los obstáculos
NUM_RECTANGLES = 5  # Número de obstáculos rectangulares
NUM_CIRCLES = 3    # Número de obstáculos circulares
MIN_OBSTACLE_SIZE = 30.0  # Tamaño mínimo de los obstáculos
MAX_OBSTACLE_SIZE = 100.0  # Tamaño máximo de los obstáculos

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

        # Espacio de acción:
        # [velocidad, ángulo_dirección]
        # Ambos son valores continuos entre -1.0 y 1.0
        self.action_space = spaces.Box(
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
            'circles': []      # Lista de [x, y, radius]
        }

    def _get_sensor_readings(self):
        """
        Calcula las lecturas de los sensores mediante ray casting.
        Detecta tanto paredes como obstáculos.
        """
        readings = np.zeros(4)
        # Ángulos de los sensores relativos al coche [frontal, trasero, izquierdo, derecho]
        sensor_angles = [0, pi, pi/2, -pi/2]
        
        for i, angle in enumerate(sensor_angles):
            # Ángulo absoluto del sensor
            abs_angle = self.state['theta'] + angle
            # Vector de dirección del sensor
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
        
        # Aplicar acción y actualizar estado
        self._calculate_kinematics(action, dt)
        
        # Obtener nueva observación
        observation = self._get_obs()
        
        # Calcular recompensa
        reward = 0.0
        
        # 1. Recompensa por movimiento
        current_speed = self.state['speed']
        speed_input, steer_input = action  # Obtener las acciones originales
        
        # Recompensa por movimiento hacia adelante y giros
        if speed_input > 0:  # Si la acción intenta ir hacia adelante
            # Recompensa base por intentar ir hacia adelante
            reward += 1.0
            
            # Recompensa adicional por velocidad efectiva hacia adelante
            if current_speed > 0:
                # Recompensa muy alta por mantener velocidad hacia adelante
                forward_reward = 4.0 * (current_speed / MAX_SPEED)
                reward += forward_reward
                
                # Recompensa adicional por combinar movimiento hacia adelante con giro
                if abs(steer_input) > 0.1:  # Si está girando mientras avanza
                    # Mayor recompensa por giros mientras avanza
                    turning_reward = 2.0 * abs(steer_input)
                    reward += turning_reward
        else:
            # Penalización muy severa por intentar retroceder
            reward -= 5.0
            
            # Penalización adicional si efectivamente está retrocediendo
            if current_speed < 0:
                reward -= 5.0 * abs(current_speed) / MAX_SPEED
        
        # 2. Manejo de obstáculos y comportamiento cerca de ellos
        sensor_frontal = observation[1]  # Lectura del sensor frontal
        sensor_izq = observation[3]      # Lectura del sensor izquierdo
        sensor_der = observation[4]      # Lectura del sensor derecho
        # Solo consideramos sensores frontales y laterales para distancias
        sensores_activos = [sensor_frontal, sensor_izq, sensor_der]
        min_sensor = np.min(sensores_activos)
        
        guarding_distance = CAR_LENGTH * 1.2  # Distancia de guarding
        danger_distance = CAR_LENGTH * 0.8    # Distancia de peligro
        optimal_distance = CAR_LENGTH * 1.0    # Distancia óptima para seguir paredes
        
        # Recompensas por proximidad controlada y comportamiento cerca de obstáculos
        if min_sensor < guarding_distance:
            if danger_distance < min_sensor < guarding_distance:
                # Recompensa GRANDE por mantener distancia óptima
                distance_to_optimal = abs(min_sensor - optimal_distance)
                proximity_reward = 8.0 * (1.0 - distance_to_optimal / (guarding_distance - danger_distance))
                reward += proximity_reward
                
                # Recompensa adicional por girar cuando está cerca de obstáculos
                if abs(steer_input) > 0.1 and current_speed > 0:
                    # Recompensa MUY ALTA por seguir paredes
                    wall_following_reward = 6.0 * abs(steer_input)
                    reward += wall_following_reward
            elif min_sensor < danger_distance:
                # Penalización moderada en zona de peligro (menor que la de retroceder)
                danger_penalty = 2.0 * ((danger_distance - min_sensor) / danger_distance)
                reward -= danger_penalty
        else:
            # Penalización por estar muy lejos de obstáculos
            reward -= 1.0
        
        # 3. Recompensa por comportamiento de navegación inteligente
        # Si detecta obstáculo al frente, recompensar fuertemente el giro apropiado
        if sensor_frontal < guarding_distance:
            # Determinar el mejor lado para girar
            if sensor_izq > sensor_der:
                # Recompensar giro a la izquierda
                if steer_input > 0:
                    reward += 4.0 * steer_input
            else:
                # Recompensar giro a la derecha
                if steer_input < 0:
                    reward += 4.0 * abs(steer_input)
        
        # Comparación de recompensas/penalizaciones:
        # - Retroceder: -5.0 a -10.0
        # - Mantener distancia óptima: hasta +8.0
        # - Seguir paredes con giros: hasta +6.0
        # - Giros apropiados cerca de obstáculos: hasta +4.0
        # - Zona de peligro: -2.0 (menos que retroceder)
        # - Lejos de obstáculos: -1.0
        
        # Verificar terminación
        terminated = False
        
        # Terminar si hay colisión con pared u obstáculo
        if min_sensor < self.car_width:
            reward -= 10.0  # Penalización severa por colisión
            terminated = True
        
        # Terminar si sale del mapa
        if not (0 <= self.state['x'] <= MAP_SIZE and 0 <= self.state['y'] <= MAP_SIZE):
            reward -= 10.0  # Penalización severa por salir del mapa
            terminated = True
            
        truncated = False
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def _generate_random_obstacles(self):
        """Genera obstáculos aleatorios en el mapa."""
        self.obstacles['rectangles'] = []
        self.obstacles['circles'] = []
        
        margin = CAR_LENGTH * 3  # Margen para evitar obstáculos muy cerca de los bordes
        
        # Generar rectángulos aleatorios
        for _ in range(NUM_RECTANGLES):
            width = self.np_random.uniform(MIN_OBSTACLE_SIZE, MAX_OBSTACLE_SIZE)
            height = self.np_random.uniform(MIN_OBSTACLE_SIZE, MAX_OBSTACLE_SIZE)
            x = self.np_random.uniform(margin, MAP_SIZE - width - margin)
            y = self.np_random.uniform(margin, MAP_SIZE - height - margin)
            self.obstacles['rectangles'].append([x, y, width, height])
        
        # Generar círculos aleatorios
        for _ in range(NUM_CIRCLES):
            radius = self.np_random.uniform(MIN_OBSTACLE_SIZE/2, MAX_OBSTACLE_SIZE/2)
            x = self.np_random.uniform(margin + radius, MAP_SIZE - radius - margin)
            y = self.np_random.uniform(margin + radius, MAP_SIZE - radius - margin)
            self.obstacles['circles'].append([x, y, radius])

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

    def reset(self, seed=None, options=None):
        """
        Reinicia el entorno a un estado inicial.
        """
        super().reset(seed=seed)
        
        # Generar nuevos obstáculos
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
        
        # Hacer el carro más rectangular (reducir el ancho)
        car_width = self.car_width * 0.7  # Reducir el ancho para hacerlo más rectangular
        
        # Puntos del coche (forma rectangular)
        car_points = [
            (-CAR_LENGTH/2, -car_width),    # Atrás izquierda
            (CAR_LENGTH/2, -car_width),     # Frente izquierda
            (CAR_LENGTH/2, car_width),      # Frente derecha
            (-CAR_LENGTH/2, car_width),     # Atrás derecha
        ]
        
        # Rotar puntos del cuerpo principal
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
        
        # Puntos para el rectángulo rojo del frente
        front_width = car_width * 0.8  # Un poco más estrecho que el cuerpo
        front_length = CAR_LENGTH * 0.2  # 20% del largo total
        front_offset = CAR_LENGTH/2 - front_length/2  # Posicionar en el frente
        
        front_points = [
            (front_offset, -front_width),                 # Izquierda atrás
            (front_offset + front_length, -front_width),  # Izquierda frente
            (front_offset + front_length, front_width),   # Derecha frente
            (front_offset, front_width),                  # Derecha atrás
        ]
        
        # Rotar puntos del frente
        rotated_front_points = []
        for x, y in front_points:
            rx = x * cos(self.state['theta']) - y * sin(self.state['theta'])
            ry = x * sin(self.state['theta']) + y * cos(self.state['theta'])
            rotated_front_points.append((
                int(car_pos[0] + rx),
                int(car_pos[1] + ry)
            ))
        
        # Dibujar el rectángulo rojo del frente
        pygame.draw.polygon(canvas, (255, 0, 0), rotated_front_points)  # Rojo
        
        # Dibujar sensores
        sensor_readings = self._get_sensor_readings()
        sensor_angles = [0, pi, pi/2, -pi/2]
        guarding_distance = CAR_LENGTH * 1.2  # Distancia de guardia
        
        for reading, angle in zip(sensor_readings, sensor_angles):
            abs_angle = self.state['theta'] + angle
            
            # Punto donde termina la zona roja (umbral de guardia)
            guard_x = car_pos[0] + cos(abs_angle) * guarding_distance
            guard_y = car_pos[1] + sin(abs_angle) * guarding_distance
            
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