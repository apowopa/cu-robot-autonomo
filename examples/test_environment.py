import gymnasium as gym
import numpy as np
import pygame
import time
import crt_car_env

def test_manual_control():
    """
    Permite controlar el carro manualmente usando las teclas:
    - Flechas Arriba/Abajo: Acelerar/Frenar
    - Flechas Izquierda/Derecha: Girar
    - R: Reiniciar el entorno
    - Q: Salir
    """
    env = gym.make('CRTCar-v0', render_mode="human")
    observation, info = env.reset()

    print("\n=== Control Manual del Carro CRT ===")
    print("Controles:")
    print("- Flecha Arriba/Abajo: Acelerar/Frenar")
    print("- Flecha Izquierda/Derecha: Girar")
    print("- R: Reiniciar entorno")
    print("- Q: Salir")
    print("\nInformación del Entorno:")
    print("- Sensores:", observation[1:])
    print("- Orientación:", observation[0])
    print("- Info:", info)

    speed = 0.0
    steering = 0.0
    running = True

    while running:
        # Procesar eventos de pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Reiniciar
                    observation, info = env.reset()
                    print("\nEntorno reiniciado!")
                    print("Nuevas lecturas de sensores:", observation[1:])
                elif event.key == pygame.K_q:  # Salir
                    running = False

        # Obtener estado de las teclas
        keys = pygame.key.get_pressed()
        
        # Actualizar velocidad y dirección basado en las teclas
        if keys[pygame.K_UP]:
            speed = min(speed + 0.1, 1.0)
        elif keys[pygame.K_DOWN]:
            speed = max(speed - 0.1, -1.0)
        else:
            speed *= 0.95  # Desaceleración suave

        if keys[pygame.K_LEFT]:
            steering = max(steering - 0.1, -1.0)
        elif keys[pygame.K_RIGHT]:
            steering = min(steering + 0.1, 1.0)
        else:
            steering *= 0.7  # Retorno suave al centro

        # Ejecutar acción
        action = np.array([speed, steering], dtype=np.float32)
        observation, reward, terminated, truncated, info = env.step(action)

        # Mostrar información relevante
        if terminated:
            print("\n¡Episodio terminado!")
            print(f"Recompensa final: {reward:.2f}")
            print("Razón: Colisión" if reward < -1 else "Razón: Objetivo alcanzado")
            observation, info = env.reset()
            speed = 0.0
            steering = 0.0

        time.sleep(0.02)  # 50 FPS aproximadamente

    env.close()

def test_autonomous_navigation():
    """
    Prueba el entorno con un comportamiento autónomo simple:
    El carro intentará evitar obstáculos girando cuando los sensores detecten algo cerca.
    """
    env = gym.make('CRTCar-v0', render_mode="human")
    observation, info = env.reset()

    print("\n=== Prueba de Navegación Autónoma Simple ===")
    print("El carro intentará evitar obstáculos automáticamente")
    print("Presiona Ctrl+C para detener")

    try:
        while True:
            # Obtener lecturas de los sensores
            _, front, back, left, right = observation

            # Lógica simple de evitación de obstáculos
            speed = 0.5  # Velocidad constante hacia adelante
            steering = 0.0

            # Si hay obstáculo cerca al frente, girar hacia el lado con más espacio
            if front < 50:
                speed = 0.2  # Reducir velocidad
                steering = 1.0 if left > right else -1.0
            elif left < 30:  # Si está muy cerca de un obstáculo a la izquierda
                steering = -0.5
            elif right < 30:  # Si está muy cerca de un obstáculo a la derecha
                steering = 0.5

            # Ejecutar acción
            action = np.array([speed, steering], dtype=np.float32)
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated:
                print("\nColisión detectada - Reiniciando entorno")
                observation, info = env.reset()

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nPrueba terminada por el usuario")

    env.close()

if __name__ == "__main__":
    print("\nSelecciona el modo de prueba:")
    print("1. Control Manual (usando el teclado)")
    print("2. Navegación Autónoma Simple")
    
    while True:
        choice = input("\nIngresa tu elección (1-2): ")
        if choice == "1":
            test_manual_control()
            break
        elif choice == "2":
            test_autonomous_navigation()
            break
        else:
            print("Opción no válida. Por favor, elige 1 o 2.")