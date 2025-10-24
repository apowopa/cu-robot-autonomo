import gymnasium as gym
import numpy as np
import torch
import wandb
from datetime import datetime
import os
from agents.dqn_agent import DQNAgent
import crt_car_env  # Importar el paquete del entorno

def discretize_action(action_idx, num_speed_levels=3, num_steering_levels=5):
    """
    Convierte un índice de acción discreto en una acción continua.
    """
    # Crear una cuadrícula de acciones posibles
    speeds = np.linspace(-1.0, 1.0, num_speed_levels)
    steerings = np.linspace(-1.0, 1.0, num_steering_levels)
    
    # Calcular los índices de velocidad y dirección
    speed_idx = action_idx // num_steering_levels
    steering_idx = action_idx % num_steering_levels
    
    return np.array([speeds[speed_idx], steerings[steering_idx]], dtype=np.float32)

def train_dqn(env_name='CRTCar-v0', 
              n_episodes=1000,
              max_t=1000,
              eps_start=1.0,
              eps_end=0.01,
              eps_decay=0.995,
              checkpoint_dir='checkpoints',
              use_wandb=True):
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
    """
    # Crear el entorno con visualización
    env = gym.make(env_name, render_mode="human")
    
    # Configurar el agente
    # Obtener el tamaño del espacio de estados del entorno real
    test_obs, _ = env.reset()
    if isinstance(test_obs, dict):
        # Si la observación es un diccionario, aplanarla
        state_size = test_obs["orientation"].shape[0] + test_obs["sensors"].shape[0]
    else:
        state_size = test_obs.shape[0]
    
    action_size = 15  # 3 niveles de velocidad * 5 niveles de dirección
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    
    # Configurar W&B
    if use_wandb:
        wandb.init(
            project="crt-car-dqn",
            config={
                "n_episodes": n_episodes,
                "max_t": max_t,
                "eps_start": eps_start,
                "eps_end": eps_end,
                "eps_decay": eps_decay,
                "buffer_size": agent.memory.memory.maxlen,
                "batch_size": agent.batch_size,
                "gamma": agent.gamma,
                "tau": agent.tau,
                "lr": agent.optimizer.param_groups[0]['lr'],
                "update_every": agent.update_every
            }
        )
    
    # Crear directorio para checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Lista para almacenar puntuaciones
    scores = []
    eps = eps_start
    
    print("Iniciando entrenamiento...")
    
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        if isinstance(state, dict):
            # Aplanar la observación si es un diccionario
            state = np.concatenate([state["orientation"], state["sensors"]])
        score = 0.0  # Inicializar como float
        
        for t in range(max_t):
            # Seleccionar acción
            action_idx = agent.act(state, eps)
            action = discretize_action(action_idx)
            
            # Ejecutar acción
            next_state, reward, terminated, truncated, _ = env.step(action)
            if isinstance(next_state, dict):
                # Aplanar la siguiente observación si es un diccionario
                next_state = np.concatenate([next_state["orientation"], next_state["sensors"]])
            done = terminated or truncated
            
            # Guardar experiencia y aprender
            agent.step(state, action_idx, reward, next_state, done)
            
            state = next_state
            score += float(reward)  # Convertir a float explícitamente
            
            if done:
                break
        
        # Actualizar epsilon
        eps = max(eps_end, eps_decay*eps)
        
        # Guardar puntuación
        scores.append(score)
        
        # Imprimir progreso
        print(f'\rEpisodio {i_episode}\tPuntuación: {score:.2f}\tEpsilon: {eps:.2f}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisodio {i_episode}\tPuntuación media: {np.mean(scores[-100:]):.2f}')
            
            # Guardar checkpoint
            checkpoint_path = os.path.join(
                checkpoint_dir, 
                f'dqn_checkpoint_episode_{i_episode}.pth'
            )
            agent.save(checkpoint_path)
        
        # Logging en W&B
        if use_wandb:
            wandb.log({
                "episode": i_episode,
                "score": score,
                "epsilon": eps,
                "average_score": np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            })
    
    # Guardar modelo final
    final_checkpoint_path = os.path.join(
        checkpoint_dir,
        f'dqn_final_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
    )
    agent.save(final_checkpoint_path)
    
    if use_wandb:
        wandb.finish()
    
    return scores, agent

if __name__ == "__main__":
    # Configuración del experimento
    config = {
        "env_name": "CRTCar-v0",  # Explicitly set the environment name
        "n_episodes": 1000,
        "max_t": 1000,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay": 0.995,
        "checkpoint_dir": "checkpoints",
        "use_wandb": False  # Desactivado para no usar W&B
    }
    
    # Entrenar el agente
    scores, agent = train_dqn(**config)