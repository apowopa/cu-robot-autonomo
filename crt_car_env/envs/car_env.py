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
        # Recompensa base por velocidad (fomenta el movimiento)
        reward = abs(self.state['speed']) / MAX_SPEED * 0.1
        
        # Penalización por estar cerca de las paredes
        min_sensor = np.min(observation[1:])  # Excluir la orientación
        if min_sensor < CAR_LENGTH:
            reward -= (CAR_LENGTH - min_sensor) / CAR_LENGTH
        
        # Verificar terminación
        terminated = False
        
        # Terminar si hay colisión con pared
        if min_sensor < self.car_width:
            reward -= 10.0  # Gran penalización por colisión
            terminated = True
        
        # Terminar si sale del mapa
        if not (0 <= self.state['x'] <= MAP_SIZE and 0 <= self.state['y'] <= MAP_SIZE):
            reward -= 5.0
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
        
        # Puntos del coche (forma rectangular)
        car_points = [
            (-CAR_LENGTH/2, -self.car_width),    # Atrás izquierda
            (CAR_LENGTH/2, -self.car_width),     # Frente izquierda
            (CAR_LENGTH/2, self.car_width),      # Frente derecha
            (-CAR_LENGTH/2, self.car_width),     # Atrás derecha
        ]
        
        # Rotar puntos
        rotated_points = []
        for x, y in car_points:
            rx = x * cos(self.state['theta']) - y * sin(self.state['theta'])
            ry = x * sin(self.state['theta']) + y * cos(self.state['theta'])
            rotated_points.append((
                int(car_pos[0] + rx),
                int(car_pos[1] + ry)
            ))
        
        # Dibujar cuerpo del coche
        pygame.draw.polygon(canvas, (0, 0, 255), rotated_points)  # Azul
        
        # Dibujar dirección (frente del coche)
        front_x = car_pos[0] + cos(self.state['theta']) * CAR_LENGTH/2
        front_y = car_pos[1] + sin(self.state['theta']) * CAR_LENGTH/2
        pygame.draw.circle(canvas, (255, 255, 0), (int(front_x), int(front_y)), 5)
        
        # Dibujar sensores
        sensor_readings = self._get_sensor_readings()
        sensor_angles = [0, pi, pi/2, -pi/2]
        for reading, angle in zip(sensor_readings, sensor_angles):
            end_x = car_pos[0] + cos(self.state['theta'] + angle) * reading
            end_y = car_pos[1] + sin(self.state['theta'] + angle) * reading
            color = (0, 255, 0) if reading > CAR_LENGTH else (255, 0, 0)  # Verde si está lejos, rojo si está cerca
            pygame.draw.line(canvas, color, car_pos, (int(end_x), int(end_y)), 2)

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