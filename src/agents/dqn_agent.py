import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from src.models.dqn import DQN, DuelingDQN
from src.memory.replay_memory import ReplayMemory, PrioritizedReplayMemory
from src.utils.config import DQNParams

logger = logging.getLogger("snake_rl")

# Agent that uses DQN for reinforcement learning
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load hyperparameters from config
        self.gamma = DQNParams.GAMMA
        self.learning_rate = DQNParams.LEARNING_RATE
        self.epsilon = DQNParams.EPSILON_START
        self.epsilon_min = DQNParams.EPSILON_MIN
        self.epsilon_decay = DQNParams.EPSILON_DECAY
        self.batch_size = DQNParams.BATCH_SIZE
        self.memory_size = DQNParams.MEMORY_SIZE
        self.target_update_freq = DQNParams.TARGET_UPDATE_FREQ
        self.use_double_dqn = DQNParams.USE_DOUBLE_DQN
        self.use_dueling_dqn = DQNParams.USE_DUELING_DQN
        self.use_prioritized_replay = DQNParams.USE_PRIORITIZED_REPLAY
        self.hidden_size = DQNParams.HIDDEN_SIZE
        self.num_layers = DQNParams.NUM_LAYERS

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

    def save_model(self, episode, score, timestamp):
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

    def plot_stats(self, timestamp, show=True, save=True):
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
