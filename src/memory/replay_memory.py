import random
import numpy as np
import torch
from collections import deque

# Experience Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)
        return (
            torch.cat(states),
            torch.tensor(actions),
            torch.cat(next_states),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.bool)
        )

    def __len__(self):
        return len(self.memory)


# Prioritized Experience Replay Memory
class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Weight exponent
        self.beta_increment = beta_increment  # For annealing beta to 1
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

    def push(self, state, action, next_state, reward, done):
        max_priority = self.max_priority if len(self.memory) > 0 else 1.0

        if len(self.memory) < self.capacity:
            self.memory.append((state, action, next_state, reward, done))
        else:
            self.memory[self.position] = (state, action, next_state, reward, done)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.memory) < self.capacity:
            probs = self.priorities[:len(self.memory)]
        else:
            probs = self.priorities

        probs = probs ** self.alpha
        probs = probs / probs.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        # Calculate importance sampling weights
        weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)

        # Increase beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        batch = list(zip(*samples))
        states = torch.cat(batch[0])
        actions = torch.tensor(batch[1])
        next_states = torch.cat(batch[2])
        rewards = torch.tensor(batch[3], dtype=torch.float32)
        dones = torch.tensor(batch[4], dtype=torch.bool)

        return states, actions, next_states, rewards, dones, indices, weights

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = error + 1e-5  # Small constant to ensure non-zero priority
            self.max_priority = max(self.max_priority, error)

    def __len__(self):
        return len(self.memory)
