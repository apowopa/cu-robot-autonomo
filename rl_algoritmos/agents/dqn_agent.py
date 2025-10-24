import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class QNetwork(nn.Module):
    """Red neuronal para aproximar la función Q."""
    
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """Buffer para almacenar y muestrear experiencias."""
    
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        
    def add(self, state, action, reward, next_state, done):
        """Añadir una nueva experiencia al buffer."""
        self.memory.append((state, action, reward, next_state, done))
        
    def sample(self):
        """Muestrear un batch de experiencias aleatorias."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).float()
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float()
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float()
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.memory)

class DQNAgent:
    """Agente que implementa Deep Q-Learning con experiencia repetida."""
    
    def __init__(
        self,
        state_size,
        action_size,
        hidden_size=64,
        buffer_size=100000,
        batch_size=64,
        gamma=0.99,
        tau=1e-3,
        learning_rate=5e-4,
        update_every=4,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma  # Factor de descuento
        self.tau = tau     # Para actualización suave
        self.update_every = update_every
        self.device = device
        
        # Redes Q
        self.qnetwork_local = QNetwork(state_size, action_size, hidden_size).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, hidden_size).to(device)
        self.optimizer = torch.optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        
        # Replay Buffer
        self.memory = ReplayBuffer(buffer_size, batch_size)
        
        # Inicializar contador de pasos
        self.t_step = 0
        
    def step(self, state, action, reward, next_state, done):
        # Guardar experiencia en replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Aprender cada update_every pasos
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
            
    def act(self, state, eps=0.):
        """Retorna acciones para un estado dado según la política actual."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        # Epsilon-greedy
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
            
    def learn(self, experiences):
        """Actualizar parámetros del modelo usando un batch de experiencias."""
        states, actions, rewards, next_states, dones = experiences
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions.long())
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Actualización suave de la red target
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        
    def soft_update(self, local_model, target_model):
        """Actualización suave del modelo target."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
            
    def save(self, filename):
        """Guardar el modelo."""
        torch.save({
            'qnetwork_local_state_dict': self.qnetwork_local.state_dict(),
            'qnetwork_target_state_dict': self.qnetwork_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename)
        
    def load(self, filename):
        """Cargar un modelo guardado."""
        checkpoint = torch.load(filename)
        self.qnetwork_local.load_state_dict(checkpoint['qnetwork_local_state_dict'])
        self.qnetwork_target.load_state_dict(checkpoint['qnetwork_target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])