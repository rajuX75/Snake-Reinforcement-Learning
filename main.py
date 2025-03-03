import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import time
import matplotlib.pyplot as plt
from collections import deque
import logging
from datetime import datetime
import json
import argparse
from pathlib import Path
import config

# Set up logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"snake_rl_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("snake_rl")


# Deep Q-Network (DQN) model
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(DQN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))

        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))

        self.output = nn.Linear(hidden_size, output_size)

        # Initialize weights
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight)
        nn.init.kaiming_normal_(self.output.weight)

    def forward(self, x):
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.1)
        return self.output(x)


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


# Snake class
class Snake:
    def __init__(self):
        self.reset()

    def reset(self):
        x = config.GameParams.GRID_WIDTH // 2
        y = config.GameParams.GRID_HEIGHT // 2
        self.body = [(x, y), (x-1, y), (x-2, y)]
        self.direction = (1, 0)  # Initial direction: right
        self.grow = False

    def move(self):
        head_x, head_y = self.body[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        if not self.grow:
            self.body.pop()
        else:
            self.grow = False

        self.body.insert(0, new_head)

    def get_head(self):
        return self.body[0]

    def check_collision(self):
        head = self.get_head()
        head_x, head_y = head

        # Check if snake hits the wall
        if (head_x < 0 or head_x >= config.GameParams.GRID_WIDTH or
            head_y < 0 or head_y >= config.GameParams.GRID_HEIGHT):
            return True

        # Check if snake hits itself
        if head in self.body[1:]:
            return True

        return False

    def change_direction(self, direction):
        # Prevent 180-degree turns
        if (self.direction[0] + direction[0] != 0 or
            self.direction[1] + direction[1] != 0):
            self.direction = direction

    def grow_snake(self):
        self.grow = True


# Game Environment
class SnakeGameEnv:
    def __init__(self, render_mode="human", fps=None, snake_speed=None):
        self.snake = Snake()
        self.food = None
        self.score = 0
        self.steps = 0
        self.max_steps = config.GameParams.MAX_STEPS
        self.render_mode = render_mode
        self.previous_distance = 0
        self.state_size = config.GameParams.STATE_SIZE  # State representation size
        self.action_size = 4  # Up, Down, Left, Right
        self.directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right
        self.steps_without_food = 0

        # Set custom FPS if provided
        self.fps = fps if fps is not None else config.GameParams.FPS

        # Set custom snake speed if provided
        self.snake_speed = snake_speed if snake_speed is not None else config.GameParams.SNAKE_SPEED

        self.frame_count = 0  # To track frames for speed control

        if render_mode == "human":
            pygame.init()
            pygame.display.set_caption("Snake Reinforcement Learning")
            self.screen = pygame.display.set_mode((config.GameParams.WIDTH, config.GameParams.HEIGHT))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, config.GameParams.FONT_SIZE)

        self.reset()

    def reset(self):
        self.snake.reset()
        self.spawn_food()
        self.score = 0
        self.steps = 0
        self.steps_without_food = 0
        self.frame_count = 0
        return self.get_state()

    def spawn_food(self):
        while True:
            x = random.randint(0, config.GameParams.GRID_WIDTH - 1)
            y = random.randint(0, config.GameParams.GRID_HEIGHT - 1)
            self.food = (x, y)

            # Make sure food doesn't spawn on snake
            if self.food not in self.snake.body:
                break

        self.previous_distance = self.get_food_distance()

    def get_food_distance(self):
        head_x, head_y = self.snake.get_head()
        food_x, food_y = self.food
        return abs(head_x - food_x) + abs(head_y - food_y)  # Manhattan distance

    def get_state(self):
        head_x, head_y = self.snake.get_head()
        food_x, food_y = self.food

        # Check if danger is straight ahead, left, or right
        direction = self.snake.direction
        dir_up = direction == (0, -1)
        dir_down = direction == (0, 1)
        dir_left = direction == (-1, 0)
        dir_right = direction == (1, 0)

        # Define points in front, left, and right of the snake
        # Look up to N steps ahead
        look_ahead = []
        for d in range(1, config.GameParams.LOOK_AHEAD + 1):
            point_straight = (head_x + direction[0] * d, head_y + direction[1] * d)
            left_dir = (-direction[1], direction[0])
            point_left = (head_x + left_dir[0] * d, head_y + left_dir[1] * d)
            right_dir = (direction[1], -direction[0])
            point_right = (head_x + right_dir[0] * d, head_y + right_dir[1] * d)

            # Check if these points are dangerous (wall or snake body)
            danger_straight = (point_straight[0] < 0 or point_straight[0] >= config.GameParams.GRID_WIDTH or
                            point_straight[1] < 0 or point_straight[1] >= config.GameParams.GRID_HEIGHT or
                            point_straight in self.snake.body)

            danger_left = (point_left[0] < 0 or point_left[0] >= config.GameParams.GRID_WIDTH or
                        point_left[1] < 0 or point_left[1] >= config.GameParams.GRID_HEIGHT or
                        point_left in self.snake.body)

            danger_right = (point_right[0] < 0 or point_right[0] >= config.GameParams.GRID_WIDTH or
                         point_right[1] < 0 or point_right[1] >= config.GameParams.GRID_HEIGHT or
                         point_right in self.snake.body)

            look_ahead.extend([danger_straight, danger_left, danger_right])

        # Calculate distance to walls in four directions
        dist_to_wall_up = head_y
        dist_to_wall_down = config.GameParams.GRID_HEIGHT - 1 - head_y
        dist_to_wall_left = head_x
        dist_to_wall_right = config.GameParams.GRID_WIDTH - 1 - head_x

        # Normalize distances to walls
        dist_to_wall_up /= config.GameParams.GRID_HEIGHT
        dist_to_wall_down /= config.GameParams.GRID_HEIGHT
        dist_to_wall_left /= config.GameParams.GRID_WIDTH
        dist_to_wall_right /= config.GameParams.GRID_WIDTH

        # Define the state
        state = [
            # Direction
            dir_up,
            dir_down,
            dir_left,
            dir_right,

            # Food direction
            food_x < head_x,  # Food is left
            food_x > head_x,  # Food is right
            food_y < head_y,  # Food is up
            food_y > head_y,  # Food is down

            # Normalized distances to walls
            dist_to_wall_up,
            dist_to_wall_down,
            dist_to_wall_left,
            dist_to_wall_right,

            # Normalized distance to food
            self.get_food_distance() / (config.GameParams.GRID_WIDTH + config.GameParams.GRID_HEIGHT),

            # Snake length (normalized)
            len(self.snake.body) / (config.GameParams.GRID_WIDTH * config.GameParams.GRID_HEIGHT),

            # Steps without food (normalized)
            self.steps_without_food / config.GameParams.MAX_STEPS_WITHOUT_FOOD
        ]

        # Add look ahead dangers
        state.extend(look_ahead)

        # Convert boolean values to int
        state = [int(i) if isinstance(i, bool) else i for i in state]

        return torch.tensor([state], dtype=torch.float32)

    def step(self, action):
        # Apply the action - only move the snake based on speed
        self.snake.change_direction(self.directions[action])

        self.frame_count += 1
        move_snake = False

        # Check if it's time to move the snake based on speed
        if self.frame_count >= self.snake_speed:
            move_snake = True
            self.frame_count = 0

        if move_snake:
            self.snake.move()
            self.steps += 1
            self.steps_without_food += 1

        # Check if the game is over
        done = False
        reward = 0

        # Check if snake collides with itself or the wall
        if self.snake.check_collision():
            reward = config.RewardParams.REWARD_DEATH
            done = True

        # Check if snake eats the food
        elif self.snake.get_head() == self.food:
            self.snake.grow_snake()
            self.score += 1
            self.steps_without_food = 0
            reward = config.RewardParams.REWARD_FOOD
            reward += config.RewardParams.REWARD_FOOD_BONUS * len(self.snake.body)  # Bonus for longer snake
            self.spawn_food()

        # Give small rewards/penalties based on moving towards/away from food
        elif move_snake:
            current_distance = self.get_food_distance()
            if current_distance < self.previous_distance:
                reward = config.RewardParams.REWARD_MOVE_TOWARDS_FOOD
            elif current_distance > self.previous_distance:
                reward = config.RewardParams.REWARD_MOVE_AWAY_FROM_FOOD
            else:
                reward = config.RewardParams.REWARD_SURVIVAL

            self.previous_distance = current_distance

            # Penalize for too many steps without food
            if self.steps_without_food > config.GameParams.MAX_STEPS_WITHOUT_FOOD:
                reward += config.RewardParams.REWARD_STUCK_PENALTY
                done = True

        # End the episode if it takes too long
        if self.steps >= self.max_steps:
            done = True

        # Get the new state
        next_state = self.get_state()

        return next_state, reward, done, self.score

    def render(self):
        if self.render_mode != "human":
            return

        self.screen.fill(config.Colors.BLACK)

        # Draw grid lines
        for x in range(0, config.GameParams.WIDTH, config.GameParams.CELL_SIZE):
            pygame.draw.line(self.screen, config.Colors.GRAY, (x, 0), (x, config.GameParams.HEIGHT), 1)
        for y in range(0, config.GameParams.HEIGHT, config.GameParams.CELL_SIZE):
            pygame.draw.line(self.screen, config.Colors.GRAY, (0, y), (config.GameParams.WIDTH, y), 1)

        # Draw food
        food_rect = pygame.Rect(
            self.food[0] * config.GameParams.CELL_SIZE,
            self.food[1] * config.GameParams.CELL_SIZE,
            config.GameParams.CELL_SIZE,
            config.GameParams.CELL_SIZE
        )
        pygame.draw.rect(self.screen, config.Colors.RED, food_rect)

        # Draw snake
        for i, (x, y) in enumerate(self.snake.body):
            snake_rect = pygame.Rect(
                x * config.GameParams.CELL_SIZE,
                y * config.GameParams.CELL_SIZE,
                config.GameParams.CELL_SIZE,
                config.GameParams.CELL_SIZE
            )

            if i == 0:  # Head
                pygame.draw.rect(self.screen, config.Colors.DARK_BLUE, snake_rect)
            else:  # Body
                color_intensity = max(50, 200 - i * 10)  # Gradient effect
                pygame.draw.rect(self.screen, (0, color_intensity, 0), snake_rect)

        # Draw score and steps
        score_text = self.font.render(f"Score: {self.score}", True, config.Colors.WHITE)
        steps_text = self.font.render(f"Steps: {self.steps}", True, config.Colors.WHITE)
        speed_text = self.font.render(f"Speed: {self.snake_speed}", True, config.Colors.WHITE)
        fps_text = self.font.render(f"FPS: {self.fps}", True, config.Colors.WHITE)

        self.screen.blit(score_text, (10, 10))
        self.screen.blit(steps_text, (10, 40))
        self.screen.blit(speed_text, (10, 70))
        self.screen.blit(fps_text, (10, 100))

        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        if self.render_mode == "human":
            pygame.quit()


# Agent that uses DQN for reinforcement learning
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load hyperparameters from config
        self.gamma = config.DQNParams.GAMMA
        self.learning_rate = config.DQNParams.LEARNING_RATE
        self.epsilon = config.DQNParams.EPSILON_START
        self.epsilon_min = config.DQNParams.EPSILON_MIN
        self.epsilon_decay = config.DQNParams.EPSILON_DECAY
        self.batch_size = config.DQNParams.BATCH_SIZE
        self.memory_size = config.DQNParams.MEMORY_SIZE
        self.target_update_freq = config.DQNParams.TARGET_UPDATE_FREQ
        self.use_double_dqn = config.DQNParams.USE_DOUBLE_DQN
        self.use_dueling_dqn = config.DQNParams.USE_DUELING_DQN
        self.use_prioritized_replay = config.DQNParams.USE_PRIORITIZED_REPLAY
        self.hidden_size = config.DQNParams.HIDDEN_SIZE
        self.num_layers = config.DQNParams.NUM_LAYERS

        # Networks
        if self.use_dueling_dqn:
            self.policy_net = DuelingDQN(state_size, self.hidden_size, action_size, self.num_layers).to(self.device)
            self.target_net = DuelingDQN(state_size, self.hidden_size, action_size, self.num_layers).to(self.device)
        else:
            self.policy_net = DQN(state_size, self.hidden_size, action_size, self.num_layers).to(self.device)
            self.target_net = DQN(state_size, self.hidden_size, action_size, self.num_layers).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        # Memory
        if self.use_prioritized_replay:
            self.memory = PrioritizedReplayMemory(self.memory_size)
        else:
            self.memory = ReplayMemory(self.memory_size)

        self.update_count = 0

        # Stats for logging
        self.losses = []
        self.scores = []
        self.average_scores = []
        self.epsilons = []

        # Create model directory
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)

        logger.info(f"DQNAgent initialized. Using device: {self.device}")
        logger.info(f"Network architecture: {self.policy_net}")
        logger.info(f"Using Double DQN: {self.use_double_dqn}")
        logger.info(f"Using Dueling DQN: {self.use_dueling_dqn}")
        logger.info(f"Using Prioritized Replay: {self.use_prioritized_replay}")

    def get_action(self, state, training=True):
        # Epsilon-greedy policy
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state = state.to(self.device)
                q_values = self.policy_net(state)
                return q_values.max(1)[1].item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        # Sample from replay memory
        if self.use_prioritized_replay:
            states, actions, next_states, rewards, dones, indices, weights = self.memory.sample(self.batch_size)
            weights = weights.to(self.device)
        else:
            states, actions, next_states, rewards, dones = self.memory.sample(self.batch_size)
            weights = torch.ones(self.batch_size, device=self.device)

        states = states.to(self.device)
        actions = actions.to(self.device)
        next_states = next_states.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Compute Q(s_{t+1}, a) for all next states
        if self.use_double_dqn:
            # Double DQN: use policy_net to select action and target_net to evaluate it
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
        else:
            next_q_values = self.target_net(next_states).max(1)[0]

        # Compute the expected Q values
        expected_q_values = rewards + self.gamma * next_q_values * (~dones)

        # Compute loss with importance sampling weights for prioritized replay
        if self.use_prioritized_replay:
            td_errors = torch.abs(q_values.squeeze() - expected_q_values).detach().cpu().numpy()
            self.memory.update_priorities(indices, td_errors)

        # Compute Huber loss
        loss = F.smooth_l1_loss(q_values.squeeze(), expected_q_values, reduction='none')
        loss = (loss * weights).mean()  # Apply importance sampling weights

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, episode, score):
        model_path = self.model_dir / f"snake_dqn_ep{episode}_score{score}_{timestamp}.pt"
        torch.save({
            'episode': episode,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'scores': self.scores,
            'losses': self.losses,
            'epsilons': self.epsilons,
            'hyperparams': {
                'gamma': self.gamma,
                'learning_rate': self.learning_rate,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay,
                'batch_size': self.batch_size,
                'memory_size': self.memory_size,
                'target_update_freq': self.target_update_freq,
                'use_double_dqn': self.use_double_dqn,
                'use_dueling_dqn': self.use_dueling_dqn,
                'use_prioritized_replay': self.use_prioritized_replay,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers
            }
        }, model_path)
        logger.info(f"Model saved to {model_path}")

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.scores = checkpoint.get('scores', [])
        self.losses = checkpoint.get('losses', [])
        self.epsilons = checkpoint.get('epsilons', [])

        # Load hyperparameters if available
        hyperparams = checkpoint.get('hyperparams', {})
        if hyperparams:
            self.gamma = hyperparams.get('gamma', self.gamma)
            self.learning_rate = hyperparams.get('learning_rate', self.learning_rate)
            self.epsilon_min = hyperparams.get('epsilon_min', self.epsilon_min)
            self.epsilon_decay = hyperparams.get('epsilon_decay', self.epsilon_decay)
            self.batch_size = hyperparams.get('batch_size', self.batch_size)
            self.target_update_freq = hyperparams.get('target_update_freq', self.target_update_freq)

        logger.info(f"Model loaded from {model_path}")
        return checkpoint.get('episode', 0)

    def plot_stats(self, show=True, save=True):
        plt.figure(figsize=(15, 15))

        # Plot scores
        plt.subplot(3, 1, 1)
        plt.plot(self.scores, label='Score')
        if len(self.average_scores) > 0:
            plt.plot(self.average_scores, label='Average Score (100 episodes)', color='red')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.title('Scores')
        plt.legend()

        # Plot losses
        plt.subplot(3, 1, 2)
        plt.plot(self.losses)
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Training Loss')

        # Plot epsilon
        plt.subplot(3, 1, 3)
        plt.plot(self.epsilons)
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.title('Exploration Rate (Epsilon)')

        plt.tight_layout()

        if save:
            plots_dir = Path("plots")
            plots_dir.mkdir(exist_ok=True)
            plt.savefig(plots_dir / f"stats_{timestamp}.png")
            logger.info(f"Stats plot saved to plots/stats_{timestamp}.png")

        if show:
            plt.show()


# Dueling DQN Network
class DuelingDQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(DuelingDQN, self).__init__()

        # Feature extraction layers
        self.feature_layers = nn.ModuleList()
        self.feature_layers.append(nn.Linear(input_size, hidden_size))

        for _ in range(num_layers - 1):
            self.feature_layers.append(nn.Linear(hidden_size, hidden_size))

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size // 2, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size // 2, output_size)
        )

        # Initialize weights
        for layer in self.feature_layers:
            nn.init.kaiming_normal_(layer.weight)

        for module in self.value_stream:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)

        for module in self.advantage_stream:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)

    def forward(self, x):
        for layer in self.feature_layers:
            x = F.leaky_relu(layer(x), 0.1)

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Combine value and advantage to get Q-values
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        return value + advantage - advantage.mean(1, keepdim=True)


def train(agent, env, num_episodes, save_freq=100, render_freq=20):
    """
    Train the DQN agent on the Snake game.

    Args:
        agent: The DQN agent
        env: The game environment
        num_episodes: Number of training episodes
        save_freq: Frequency to save the model (episodes)
        render_freq: Frequency to render the game (episodes)
    """
    total_steps = 0
    best_score = 0
    episode_scores = []

    logger.info("Starting training...")
    logger.info(f"Training for {num_episodes} episodes")

    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        score = 0
        steps = 0
        episode_loss = 0

        while not done:
            # Get action from agent
            action = agent.get_action(state)

            # Take step in environment
            next_state, reward, done, episode_score = env.step(action)
            score = episode_score

            # Store transition in replay memory
            agent.memory.push(state, action, next_state, reward, done)

            # Learn from memory
            loss = agent.learn()
            if loss:
                episode_loss += loss
                agent.losses.append(loss)

            # Update state
            state = next_state
            steps += 1
            total_steps += 1

            # Render the game if needed
            if episode % render_freq == 0:
                env.render()

            # Handle exit events if rendering
            if env.render_mode == "human":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        agent.plot_stats(show=False, save=True)
                        return

        # Update exploration rate
        agent.update_epsilon()

        # Record statistics
        agent.scores.append(score)
        agent.epsilons.append(agent.epsilon)

        # Calculate running average
        episode_scores.append(score)
        if len(episode_scores) > 100:
            episode_scores.pop(0)
        avg_score = sum(episode_scores) / len(episode_scores)
        agent.average_scores.append(avg_score)

        # Log progress
        if episode % 10 == 0:
            elapsed_time = time.time() - start_time
            logger.info(f"Episode: {episode}/{num_episodes}, Score: {score}, Avg Score: {avg_score:.2f}, "
                        f"Steps: {steps}, Epsilon: {agent.epsilon:.4f}, "
                        f"Loss: {episode_loss/steps if steps > 0 else 0:.6f}, "
                        f"Time: {elapsed_time:.2f}s")

        # Save model
        if episode % save_freq == 0:
            agent.save_model(episode, score)

        # Save best model
        if score > best_score:
            best_score = score
            agent.save_model(episode, score)
            logger.info(f"New best score: {best_score} (Episode {episode})")

    # Final save
    agent.save_model(num_episodes, score)

    # Plot statistics
    agent.plot_stats(show=False, save=True)

    logger.info(f"Training completed. Best score: {best_score}")

    return agent


def main():
    """Main function to run the Snake RL training or play with a trained agent."""
    parser = argparse.ArgumentParser(description='Snake Reinforcement Learning')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'play'],
                        help='Train model or play with trained model')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes for training')
    parser.add_argument('--load', type=str, help='Path to pre-trained model to load')
    parser.add_argument('--render', action='store_true', help='Enable rendering for all episodes')
    parser.add_argument('--render_freq', type=int, default=20,
                        help='Frequency of episodes to render during training')
    parser.add_argument('--save_freq', type=int, default=100,
                        help='Frequency of episodes to save model during training')
    parser.add_argument('--fps', type=int, default=config.GameParams.FPS, help='FPS for rendering')
    parser.add_argument('--speed', type=int, default=config.GameParams.SNAKE_SPEED,
                        help='Snake speed (lower is faster)')

    args = parser.parse_args()

    # Set rendering mode based on arguments
    render_mode = "human" if args.mode == 'play' or args.render else "none"

    # Create environment
    env = SnakeGameEnv(render_mode=render_mode, fps=args.fps, snake_speed=args.speed)

    # Create agent
    agent = DQNAgent(env.state_size, env.action_size)

    # Load model if specified
    start_episode = 0
    if args.load:
        start_episode = agent.load_model(args.load)

    if args.mode == 'train':
        # Train the agent
        train(agent, env, args.episodes, save_freq=args.save_freq, render_freq=args.render_freq)
    elif args.mode == 'play':
        # Play with the trained agent
        for episode in range(1, 100):  # Play up to 100 episodes
            state = env.reset()
            done = False
            score = 0

            while not done:
                # Get action from agent (no exploration)
                action = agent.get_action(state, training=False)

                # Take step in environment
                next_state, reward, done, episode_score = env.step(action)
                score = episode_score

                # Update state
                state = next_state

                # Render the game
                env.render()

                # Handle exit events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return

            logger.info(f"Episode {episode} finished with score {score}")

    # Close environment
    env.close()


if __name__ == "__main__":
    main()
