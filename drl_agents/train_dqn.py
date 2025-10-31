import gymnasium as gym
import numpy as np
import torch
from datetime import datetime
import os
import signal
import sys
from agents.dqn_agent import DQNAgent
import crt_car_env  # Importar el paquete del entorno

current_agent = None
checkpoint_dir = "checkpoints"


def signal_handler(sig, frame):
    print("\nInterrumpiendo entrenamiento. Guardando checkpoint...")
    if current_agent is not None and "args" in globals():  # Verificar que args exista
        # Crear directorio si no existe
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Guardar checkpoint de interrupción
        interrupt_checkpoint_path = os.path.join(
            checkpoint_dir,
            f"dqn_{args.tag}_interrupt_{datetime.now().strftime('%Y%m%d_%H%M')}.pth",
        )
        current_agent.save(interrupt_checkpoint_path)
        print(f"Checkpoint guardado en: {interrupt_checkpoint_path}")

    sys.exit(0)


# Registrar el manejador de señales
signal.signal(signal.SIGINT, signal_handler)


def train_dqn(
    env_name="CRTCar-v0",
    n_episodes=1000,
    max_t=1000,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.995,
    checkpoint_dir="checkpoints",
    agent=None,
    render_mode="human",
    tag="default",
    debug=False,
):
    global current_agent  # Declarar uso de variable global
    print(f"Creating environment: {env_name}")
    """
    Entrenamiento del agente DQN.
    
    Parámetros:
        env_name: Nombre del entorno de Gymnasium
        n_episodes: Número máximo de episodios de entrenamiento
        max_t: Número máximo de pasos por episodio
        eps_start: Valor inicial de epsilon para exploración
        eps_end: Valor mínimo de epsilon
        eps_decay: Factor de decaimiento de epsilon
        checkpoint_dir: Directorio para guardar los checkpoints
        use_wandb: Si se debe usar Weights & Biases para seguimiento
        agent: Agente pre-entrenado (opcional)
    """
    current_agent = agent

    # Crear directorio para checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Lista para almacenar puntuaciones
    scores = []
    eps = eps_start

    print("Iniciando entrenamiento...")

    # Crear el entorno de entrenamiento
    env = gym.make(env_name, render_mode=render_mode)
    print(f"Render mode: {render_mode}")

    # Variables para el seguimiento del rendimiento
    best_score = float("-inf")  # Mejor puntuación hasta ahora
    best_model_state = None  # Estado del mejor modelo
    no_improvement_count = 0  # Contador de episodios sin mejora
    improvement_threshold = 0.1  # Umbral de mejora (10%)
    patience = 50  # Número de episodios a esperar antes de restaurar
    evaluation_window = 10  # Ventana de episodios para calcular la media

    if debug:
        print("\n=== MODO DEBUG ACTIVADO ===")
        print(f"Estado del agente:")
        print(f"- Red neuronal: {agent.qnetwork_local}")
        print(f"- Tamaño del buffer de memoria: {agent.memory.memory.maxlen}")
        print(f"- Batch size: {agent.batch_size}")
        print(f"- Gamma: {agent.gamma}")
        print(f"- Tau: {agent.tau}")
        print(f"- Learning rate: {agent.optimizer.param_groups[0]['lr']}")
        print("\nPresiona Enter para continuar...")
        input()

    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        if isinstance(state, dict):
            # Aplanar la observación si es un diccionario
            state = np.concatenate([state["orientation"], state["sensors"]])
        score = 0.0  # Inicializar como float

        for t in range(max_t):
            # Seleccionar acción
            action_idx = agent.act(state, eps)

            # Ejecutar acción discreta directamente
            next_state, reward, terminated, truncated, info = env.step(action_idx)
            if isinstance(next_state, dict):
                # Aplanar la siguiente observación si es un diccionario
                next_state = np.concatenate(
                    [next_state["orientation"], next_state["sensors"]]
                )
            done = terminated or truncated

            # Log detallado solo en modo human y cada 50 pasos o cuando hay eventos importantes
            if render_mode == "human" and (t % 50 == 0 or float(reward) > 5.0 or done):
                orientation = state[0]  # Primer elemento es la orientación
                sensors = state[1:5]  # Los siguientes 4 son lecturas de sensores

                print(f"\n--- Estado del Agente (Ep {i_episode}, Paso {t}) ---")
                print(f"Sensores [F,T,I,D]: {sensors}")
                print(
                    f"Orientación: {orientation:.2f} rad ({np.degrees(orientation):.1f}°)"
                )
                print(f"Acción Discreta: #{action_idx}")
                print(f"Recompensa: {reward:.2f}")
                if done:
                    print(
                        f"Episodio terminado por {'(Colisión)' if terminated else '(Tiempo)'}"
                    )
                print("-" * 50)

            # Guardar experiencia y aprender
            agent.step(state, action_idx, reward, next_state, done)

            state = next_state
            score += float(reward)  # Convertir a float explícitamente

            if done:
                break

        # Actualizar epsilon
        eps = max(eps_end, eps_decay * eps)

        # Guardar puntuación
        scores.append(score)

        # Calcular puntuación media de la ventana actual
        current_window = (
            scores[-evaluation_window:] if len(scores) >= evaluation_window else scores
        )
        current_avg_score = np.mean(current_window)

        # Verificar si hay mejora
        if current_avg_score > best_score * (1 + improvement_threshold):
            # Hay una mejora significativa
            best_score = current_avg_score
            best_model_state = agent.get_model_state()
            no_improvement_count = 0
            if debug:
                print(f"\n¡Nuevo mejor modelo! Score: {current_avg_score:.2f}")
        else:
            no_improvement_count += 1

            # Si no hay mejora después de 'patience' episodios, restaurar al mejor modelo
            if no_improvement_count >= patience and best_model_state is not None:
                print(
                    f"\nNo hay mejora en {patience} episodios. Restaurando mejor modelo..."
                )
                agent.set_model_state(best_model_state)
                no_improvement_count = 0
                if debug:
                    print(f"Modelo restaurado. Mejor score: {best_score:.2f}")

        # Imprimir progreso
        if render_mode == "human":
            print(
                f"\rEpisodio {i_episode}\tPuntuación: {score:.2f}\tMejor: {best_score:.2f}\tEpsilon: {eps:.2f}",
                end="",
            )
            if i_episode % 100 == 0:
                print(
                    f"\rEpisodio {i_episode}\tPuntuación media: {current_avg_score:.2f}"
                )
        else:
            # En modo no visual, imprimir actualizaciones más concisas
            if i_episode % 10 == 0:  # Actualizar cada 10 episodios
                print(
                    f"\rEp: {i_episode}/{n_episodes} | Score: {score:.1f} | Avg: {current_avg_score:.1f} | Best: {best_score:.1f} | ε: {eps:.2f}",
                    end="",
                )

            # Guardar checkpoint cada 250 episodios
            if i_episode % 250 == 0:
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"dqn_{tag}_episode_{i_episode}.pth"
                )
                agent.save(checkpoint_path)
                print(f"\nGuardando checkpoint en episodio {i_episode}")

    # Guardar modelo final
    final_checkpoint_path = os.path.join(
        checkpoint_dir,
        f"dqn_{tag}_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth",
    )
    agent.save(final_checkpoint_path)

    # Cerrar el entorno
    env.close()

    return scores, agent


if __name__ == "__main__":
    import argparse

    # Configurar el parser de argumentos
    parser = argparse.ArgumentParser(description="Entrenamiento del agente DQN")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Ruta al archivo de checkpoint para continuar el entrenamiento",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="default",
        help="Etiqueta para identificar al agente en los checkpoints",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Número de episodios de entrenamiento",
    )
    parser.add_argument(
        "--epsilon", type=float, default=1.0, help="Valor inicial de epsilon"
    )
    parser.add_argument(
        "--render", action="store_true", help="Activar visualización del entrenamiento"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activa modo debug con pausas para verificación",
    )
    args = parser.parse_args()

    # Configuración del experimento
    config = {
        "env_name": "CRTCar-v0",
        "n_episodes": args.episodes,
        "max_t": 1000,
        "eps_start": args.epsilon,  # Usar el epsilon proporcionado
        "eps_end": 0.01,
        "eps_decay": 0.995,
        "checkpoint_dir": "checkpoints",
        "render_mode": "human"
        if args.render
        else None,  # Modo de renderizado según el argumento
        "tag": args.tag,  # Agregar el tag a la configuración
        "debug": args.debug,  # Modo debug
    }

    # Crear el entorno para obtener los tamaños de estado y acción
    env = gym.make(config["env_name"], render_mode=None)
    test_obs, _ = env.reset(seed=42)
    if isinstance(test_obs, dict):
        state_size = test_obs["orientation"].shape[0] + test_obs["sensors"].shape[0]
    else:
        state_size = test_obs.shape[0]
    action_size = env.action_space.n
    env.close()  # Cerrar el entorno temporal

    # Crear el agente
    agent = DQNAgent(state_size=state_size, action_size=action_size)

    # Actualizar la configuración para incluir el agente
    config["agent"] = agent

    # Función para encontrar el checkpoint más reciente con un tag
    def find_latest_checkpoint(tag):
        if not os.path.exists(config["checkpoint_dir"]):
            return None

        checkpoints = []
        for file in os.listdir(config["checkpoint_dir"]):
            if file.startswith(f"dqn_{tag}_") and file.endswith(".pth"):
                path = os.path.join(config["checkpoint_dir"], file)
                checkpoints.append((path, os.path.getmtime(path)))

        if not checkpoints:
            return None

        # Ordenar por tiempo de modificación y obtener el más reciente
        latest_checkpoint = sorted(checkpoints, key=lambda x: x[1], reverse=True)[0][0]
        return latest_checkpoint

    # Cargar checkpoint
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        checkpoint_path = find_latest_checkpoint(args.tag)

    if checkpoint_path:
        try:
            print(f"Cargando checkpoint: {checkpoint_path}")
            agent.load(checkpoint_path)
            print("Checkpoint cargado exitosamente")

            if args.debug:
                print("\n=== VERIFICACIÓN DE CHECKPOINT ===")
                print("Información del modelo cargado:")
                print(f"- Archivo: {checkpoint_path}")
                print(
                    f"- Redes disponibles: {[name for name, _ in agent.qnetwork_local.named_parameters()]}"
                )
                print("\nPresiona Enter para continuar con el entrenamiento...")
                input()

        except Exception as e:
            print(f"Error al cargar el checkpoint: {e}")
            exit(1)
    else:
        print(
            f"No se encontró ningún checkpoint con el tag '{args.tag}'. Iniciando nuevo entrenamiento."
        )

    # Entrenar el agente
    current_agent = agent  # Para el manejador de señales
    scores, agent = train_dqn(**config)

