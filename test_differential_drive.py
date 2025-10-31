#!/usr/bin/env python3
"""
Script de prueba para verificar el funcionamiento del modelo de tracción diferencial.
"""

import gymnasium as gym
import numpy as np
import crt_car_env


def test_differential_drive():
    """Prueba básica del entorno con tracción diferencial."""
    print("=" * 60)
    print("Prueba de Tracción Diferencial")
    print("=" * 60)
    
    # Crear el entorno
    env = gym.make("CRTCar-v0", render_mode=None)
    
    # Acceder al entorno base (sin wrappers)
    base_env = env.unwrapped
    
    print("\n1. Información del entorno:")
    print(f"   - Espacio de observación: {env.observation_space}")
    print(f"   - Espacio de acción: {env.action_space}")
    print(f"   - Número de acciones disponibles: {env.action_space.n}")
    
    print("\n2. Acciones disponibles:")
    for i in range(env.action_space.n):
        action_obj = base_env.discrete_actions.get_action(i)
        description = base_env.discrete_actions.describe_action(action_obj)
        print(f"   Acción {i}: {description}")
        print(f"      - Rueda izquierda: {action_obj.speed:.2f}")
        print(f"      - Rueda derecha: {action_obj.steering:.2f}")
    
    print("\n3. Ejecutando episodio de prueba...")
    
    # Reset del entorno
    observation, info = env.reset(seed=42)
    print(f"\n   Observación inicial: {observation}")
    print(f"   Posición inicial: ({info['x_position']:.2f}, {info['y_position']:.2f})")
    print(f"   Orientación inicial: {info['orientation']:.2f} rad")
    
    # Ejecutar algunos pasos
    total_reward = 0
    num_steps = 10
    
    print(f"\n   Ejecutando {num_steps} pasos aleatorios...")
    for step in range(num_steps):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        action_desc = base_env.discrete_actions.describe_action(
            base_env.discrete_actions.get_action(action)
        )
        
        print(f"\n   Paso {step + 1}:")
        print(f"      - Acción: {action_desc}")
        print(f"      - Recompensa: {reward:.2f}")
        print(f"      - Posición: ({info['x_position']:.2f}, {info['y_position']:.2f})")
        print(f"      - Velocidad: {info['speed']:.2f}")
        print(f"      - Sensores: F={observation[1]:.1f}, T={observation[2]:.1f}, "
              f"I={observation[3]:.1f}, D={observation[4]:.1f}")
        
        if terminated or truncated:
            print(f"\n   ¡Episodio terminado en el paso {step + 1}!")
            break
    
    print(f"\n   Recompensa total: {total_reward:.2f}")
    
    print("\n4. Probando acciones específicas...")
    
    # Reset para pruebas específicas
    observation, info = env.reset(seed=123)
    
    # Probar cada tipo de acción
    test_actions = [0, 1, 2, 3, 4, 5]  # Todas las acciones
    action_names = [
        "ADELANTE",
        "REVERSA", 
        "GIRO_IZQUIERDA",
        "GIRO_DERECHA",
        "GIRO_CERRADO_IZQUIERDA",
        "GIRO_CERRADO_DERECHA"
    ]
    
    for action_idx, action_name in zip(test_actions, action_names):
        # Ejecutar la acción 3 veces para ver el efecto
        print(f"\n   Probando {action_name}...")
        for _ in range(3):
            observation, reward, terminated, truncated, info = env.step(action_idx)
            if terminated or truncated:
                break
        
        print(f"      - Posición final: ({info['x_position']:.2f}, {info['y_position']:.2f})")
        print(f"      - Orientación final: {info['orientation']:.2f} rad")
        print(f"      - Velocidad final: {info['speed']:.2f}")
    
    env.close()
    
    print("\n" + "=" * 60)
    print("✓ Prueba completada exitosamente!")
    print("=" * 60)


if __name__ == "__main__":
    test_differential_drive()
