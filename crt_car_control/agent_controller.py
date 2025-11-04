"""
Controlador del agente para el carrito con tracción diferencial.
Este módulo integra el agente DQN entrenado con el hardware del robot.
"""

import sys
import time
from pathlib import Path

import board
import busio
import numpy as np
import torch
from adafruit_vl53l0x import VL53L0X
from gpiozero import Device, DigitalOutputDevice
from gpiozero.pins.lgpio import LGPIOFactory

# Agregar el directorio raíz al path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

# Importar el agente y el espacio de acciones
from crt_car_env.envs.actions import DiscreteActionSpace
from drl_agents.agents.dqn_agent import DQNAgent


class RobotController:
    """Controlador del robot que integra el agente DQN con el hardware."""
    
    def __init__(self, checkpoint_path: str, obstacle_distance: int = 200):
        """
        Inicializa el controlador del robot.
        
        Args:
            checkpoint_path: Ruta al checkpoint del modelo entrenado
            obstacle_distance: Distancia mínima de obstáculo en mm
        """
        self.obstacle_distance = obstacle_distance
        self.action_space = DiscreteActionSpace()
        
        # Configuración del hardware
        self._setup_hardware()
        
        # Cargar el agente
        self._load_agent(checkpoint_path)
        
    def _setup_hardware(self):
        """Configura los pines GPIO y sensores."""
        print("Configurando hardware...")
        
        # Configuración de pines
        Device.pin_factory = LGPIOFactory()
        
        # Configuración de motores (tracción diferencial)
        self.motor_left_a = DigitalOutputDevice(5)
        self.motor_left_b = DigitalOutputDevice(6)
        self.motor_right_a = DigitalOutputDevice(13)
        self.motor_right_b = DigitalOutputDevice(19)
        
        # Configuración de sensores VL53L0X
        i2c = busio.I2C(board.SCL, board.SDA)
        
        xshut_pins = {
            'frontal': 4,
            'derecha': 17,
            'izquierda': 22
        }
        
        shutdown_pins = {}
        for name, pin in xshut_pins.items():
            shutdown_pins[name] = DigitalOutputDevice(pin, initial_value=False)
        
        new_addresses = {
            'izquierda': 0x30,
            'frontal': 0x31,
            'derecha': 0x32
        }
        
        self.sensors = {}
        print("Inicializando sensores...")
        
        for name, pin_device in shutdown_pins.items():
            pin_device.on()
            time.sleep(0.1)
            
            try:
                sensor_temp = VL53L0X(i2c)
                new_addr = new_addresses[name]
                sensor_temp.set_address(new_addr)
                self.sensors[name] = sensor_temp
                print(f"Sensor '{name}' inicializado en {hex(new_addr)}")
            except Exception as e:
                print(f"Error con sensor '{name}': {e}")
                pin_device.off()
            
            time.sleep(0.05)
        
        self.shutdown_pins = shutdown_pins
        print("Hardware configurado correctamente.")
    
    def _load_agent(self, checkpoint_path: str):
        """Carga el agente DQN desde un checkpoint."""
        print(f"Cargando agente desde {checkpoint_path}...")
        
        # El estado siempre tiene 5 elementos: [theta, sensor_front, sensor_rear, sensor_left, sensor_right]
        state_size = 5
        action_size = self.action_space.n
        
        # Crear el agente con las dimensiones correctas
        self.agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            hidden_size=64,
            buffer_size=1000,
            batch_size=32,
            gamma=0.99,
            learning_rate=0.001
        )
        
        # Cargar los pesos del checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.agent.qnetwork_local.load_state_dict(checkpoint['local'])
        self.agent.qnetwork_target.load_state_dict(checkpoint['target'])
        self.agent.qnetwork_local.eval()
        
        print(f"Agente cargado: state_size={state_size}, action_size={action_size}")
    
    def read_sensors(self) -> dict:
        """
        Lee las distancias de los sensores en mm (VL53L0X.range devuelve mm).
        Conviertir automáticamente a cm para que coincida con el simulador.
        """
        distances = {}
        for name, sensor in self.sensors.items():
            try:
                # VL53L0X.range devuelve la distancia en mm
                dist_mm = sensor.range
                if dist_mm is not None:
                    # Convertir a cm para coincidir con simulador (cm)
                    distances[name] = dist_mm / 10.0
                else:
                    # Lectura None = fuera de rango = distancia máxima
                    distances[name] = 2000.0  # 200 cm (rango máximo del VL53L0X)
            except (OSError, RuntimeError, TimeoutError):
                # Timeout o error I2C - usar valor seguro por defecto (máxima distancia)
                distances[name] = 2000.0
        return distances
    
    def get_state(self) -> np.ndarray:
        """
        Construye el estado actual del robot para el agente.
        Exactamente: [theta, sensor_front, sensor_rear, sensor_left, sensor_right]
        donde distancias están en cm [0-80] como en el simulador.
        """
        if not hasattr(self, 'cached_distances'):
            self.cached_distances = self.read_sensors()
        
        distances = self.cached_distances
        
        # Distancias en cm (ya convertidas de mm en read_sensors)
        dist_front = np.clip(distances.get('frontal', 200.0), 0.0, 80.0)
        dist_left = np.clip(distances.get('izquierda', 200.0), 0.0, 80.0)
        dist_right = np.clip(distances.get('derecha', 200.0), 0.0, 80.0)
        
        # Estado: [theta, sensor_frontal, sensor_trasero, sensor_izq, sensor_der]
        state = np.array([
            0.0,          # theta - no disponible en hardware real
            dist_front,   # sensor frontal
            dist_right,   # sensor trasero (no disponemos, usar derecho como fallback)
            dist_left,    # sensor izquierdo
            dist_right    # sensor derecho
        ], dtype=np.float32)
        
        return state
    
    def execute_action(self, action_idx: int | np.intp):
        """
        Ejecuta una acción en el hardware del robot (sin prints en loop).
        
        Args:
            action_idx: Índice de la acción a ejecutar
        """
        action = self.action_space.get_action(int(action_idx))
        
        # En el modelo de tracción diferencial:
        # speed = velocidad rueda izquierda
        # steering = velocidad rueda derecha
        left_wheel = action.speed
        right_wheel = action.steering
        
        # Controlar rueda izquierda (sin print para reducir latencia)
        if left_wheel > 0.1:
            self.motor_left_a.on()
            self.motor_left_b.off()
        elif left_wheel < -0.1:
            self.motor_left_a.off()
            self.motor_left_b.on()
        else:
            self.motor_left_a.off()
            self.motor_left_b.off()
        
        # Controlar rueda derecha
        if right_wheel > 0.1:
            self.motor_right_a.on()
            self.motor_right_b.off()
        elif right_wheel < -0.1:
            self.motor_right_a.off()
            self.motor_right_b.on()
        else:
            self.motor_right_a.off()
            self.motor_right_b.off()
    
    def stop(self):
        """Detiene todos los motores."""
        self.motor_left_a.off()
        self.motor_left_b.off()
        self.motor_right_a.off()
        self.motor_right_b.off()
        print("STOP")
    
    def brake(self):
        """Frena activamente todos los motores."""
        self.motor_left_a.on()
        self.motor_left_b.on()
        self.motor_right_a.on()
        self.motor_right_b.on()
        print("BRAKE")
    
    def run(self, duration: float | None = None):
        """
        Ejecuta el controlador del robot (optimizado).
        
        Args:
            duration: Duración en segundos (None para indefinido)
        """
        print("Iniciando controlador del robot...")
        start_time = time.time()
        step_counter = 0
        
        try:
            while True:
                # Leer el estado actual
                self.cached_distances = self.read_sensors()
                state = self.get_state()
                
                # Obtener la acción del agente (sin numpy cast para reducir overhead)
                with torch.no_grad():
                    action_idx = self.agent.act(state)
                
                # Ejecutar la acción
                self.execute_action(action_idx)
                
                # Mostrar información cada N pasos (no cada iteración)
                step_counter += 1
                if step_counter % 5 == 0:  # Mostrar cada 5 pasos
                    distances = self.cached_distances
                    print(f"Step {step_counter}: F={distances.get('frontal', 0):4d}mm | "
                          f"I={distances.get('izquierda', 0):4d}mm | "
                          f"D={distances.get('derecha', 0):4d}mm")
                
                # Reducir sleep para mejores tiempos de reacción
                time.sleep(0.05)  # 50ms = 20 Hz (era 0.2 = 5 Hz)
                
                # Verificar duración
                if duration is not None and (time.time() - start_time) >= duration:
                    break
        
        except KeyboardInterrupt:
            print("\nInterrupción del usuario...")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Limpia los recursos del hardware."""
        print("Limpiando recursos...")
        self.stop()
        for pin_device in self.shutdown_pins.values():
            pin_device.off()
        print("Sistema detenido correctamente.")


def main():
    """Función principal para ejecutar el controlador."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Controlador del robot con agente DQN')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Ruta al checkpoint del modelo')
    parser.add_argument('--obstacle-distance', type=int, default=200,
                        help='Distancia mínima de obstáculo en mm (default: 200)')
    parser.add_argument('--duration', type=float, default=None,
                        help='Duración de la ejecución en segundos (default: indefinido)')
    
    args = parser.parse_args()
    
    # Crear y ejecutar el controlador
    controller = RobotController(
        checkpoint_path=args.checkpoint,
        obstacle_distance=args.obstacle_distance
    )
    
    controller.run(duration=args.duration)


if __name__ == '__main__':
    main()
