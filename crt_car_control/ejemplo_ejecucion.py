#!/usr/bin/env python3
"""
Ejemplo de cómo ejecutar el controlador del robot con el agente DQN.
Este script muestra diferentes formas de usar el RobotController.
"""

from agent_controller import RobotController


def ejemplo_basico():
    """Ejemplo básico: ejecutar el robot indefinidamente."""
    print("=" * 60)
    print("EJEMPLO 1: Ejecución básica indefinida")
    print("=" * 60)
    print("El robot se ejecutará hasta presionar Ctrl+C\n")
    
    # Crear el controlador con un modelo entrenado
    controller = RobotController(
        checkpoint_path='../checkpoints/dqn_gigi.pth',
        obstacle_distance=200  # 20cm de distancia mínima
    )
    
    # Ejecutar indefinidamente (presiona Ctrl+C para detener)
    controller.run()


def ejemplo_con_duracion():
    """Ejemplo con duración limitada."""
    print("=" * 60)
    print("EJEMPLO 2: Ejecución con duración limitada (30 segundos)")
    print("=" * 60)
    
    # Crear el controlador
    controller = RobotController(
        checkpoint_path='../checkpoints/dqn_gigi.pth',
        obstacle_distance=200
    )
    
    # Ejecutar por 30 segundos
    print("El robot se ejecutará por 30 segundos...\n")
    controller.run(duration=30.0)
    
    print("\n✓ Ejecución completada")


def ejemplo_distancia_personalizada():
    """Ejemplo con distancia de obstáculo personalizada."""
    print("=" * 60)
    print("EJEMPLO 3: Distancia de obstáculo personalizada")
    print("=" * 60)
    
    # Crear el controlador con distancia de obstáculo mayor
    controller = RobotController(
        checkpoint_path='../checkpoints/dqn_gura.pth',
        obstacle_distance=300  # 30cm de distancia mínima (más conservador)
    )
    
    print("El robot mantendrá 30cm de distancia mínima a obstáculos")
    print("Ejecutando por 60 segundos...\n")
    
    controller.run(duration=60.0)


def ejemplo_con_manejo_errores():
    """Ejemplo con manejo robusto de errores."""
    print("=" * 60)
    print("EJEMPLO 4: Ejecución con manejo de errores")
    print("=" * 60)
    
    try:
        # Crear el controlador
        controller = RobotController(
            checkpoint_path='../checkpoints/dqn_sora.pth',
            obstacle_distance=200
        )
        
        print("Robot iniciado correctamente")
        print("Presiona Ctrl+C para detener en cualquier momento\n")
        
        # Ejecutar el robot
        controller.run()
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: No se encontró el archivo del modelo")
        print(f"   {e}")
        print("\nAsegúrate de que el checkpoint existe en la ruta especificada")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupción del usuario detectada")
        print("Deteniendo el robot de forma segura...")
        
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        print("El sistema se detendrá de forma segura")
        
    finally:
        print("\n✓ Sistema detenido correctamente")


def ejemplo_test_sensores():
    """Ejemplo para probar solo los sensores sin mover el robot."""
    print("=" * 60)
    print("EJEMPLO 5: Test de sensores (sin movimiento)")
    print("=" * 60)
    
    import time
    
    # Crear el controlador
    controller = RobotController(
        checkpoint_path='../checkpoints/dqn_gigi.pth',
        obstacle_distance=200
    )
    
    print("Leyendo sensores por 10 segundos (el robot NO se moverá)...\n")
    
    try:
        for i in range(50):  # 50 lecturas = 10 segundos
            # Leer sensores
            distances = controller.read_sensors()
            
            # Mostrar información
            print(f"Lectura {i+1:2d} | "
                  f"Frontal: {distances.get('frontal', 0):4d}mm | "
                  f"Izquierda: {distances.get('izquierda', 0):4d}mm | "
                  f"Derecha: {distances.get('derecha', 0):4d}mm")
            
            time.sleep(0.2)
    
    except KeyboardInterrupt:
        print("\n\nTest interrumpido")
    
    finally:
        controller.cleanup()
        print("\n✓ Test completado")


def menu_principal():
    """Menú interactivo para seleccionar ejemplos."""
    print("\n" + "=" * 60)
    print("EJEMPLOS DE USO DEL CONTROLADOR DEL ROBOT")
    print("=" * 60)
    print("\nSelecciona un ejemplo:")
    print("  1. Ejecución básica (indefinida)")
    print("  2. Ejecución con duración limitada (30s)")
    print("  3. Distancia de obstáculo personalizada")
    print("  4. Ejecución con manejo de errores")
    print("  5. Test de sensores (sin movimiento)")
    print("  0. Salir")
    print()
    
    while True:
        try:
            opcion = input("Ingresa el número de ejemplo (0-5): ").strip()
            
            if opcion == "0":
                print("\n¡Hasta luego!")
                break
            elif opcion == "1":
                ejemplo_basico()
                break
            elif opcion == "2":
                ejemplo_con_duracion()
                break
            elif opcion == "3":
                ejemplo_distancia_personalizada()
                break
            elif opcion == "4":
                ejemplo_con_manejo_errores()
                break
            elif opcion == "5":
                ejemplo_test_sensores()
                break
            else:
                print("❌ Opción inválida. Intenta de nuevo.\n")
        
        except KeyboardInterrupt:
            print("\n\n¡Hasta luego!")
            break


if __name__ == '__main__':
    # Puedes ejecutar el menú o un ejemplo específico
    
    # Opción 1: Menú interactivo
    menu_principal()
    
    # Opción 2: Ejecutar un ejemplo específico directamente
    # Descomenta la línea del ejemplo que quieras ejecutar:
    
    # ejemplo_basico()
    # ejemplo_con_duracion()
    # ejemplo_distancia_personalizada()
    # ejemplo_con_manejo_errores()
    # ejemplo_test_sensores()
