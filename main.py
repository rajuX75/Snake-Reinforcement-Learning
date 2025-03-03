import argparse
from pathlib import Path
from datetime import datetime
import pygame

from src.environment.game_env import SnakeGameEnv
from src.agents.dqn_agent import DQNAgent
from src.training.trainer import train, play
from src.utils.logger import setup_logger
from src.utils.config import GameParams

def main():
    """Main function to run the Snake RL training or play with a trained agent."""
    # Set up logger
    logger, timestamp = setup_logger()

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
    parser.add_argument('--fps', type=int, default=GameParams.FPS, help='FPS for rendering')
    parser.add_argument('--speed', type=int, default=GameParams.SNAKE_SPEED,
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
        train(agent, env, args.episodes, timestamp, save_freq=args.save_freq, render_freq=args.render_freq)
    elif args.mode == 'play':
        # Play with the trained agent
        play(agent, env)

    # Close environment
    env.close()


if __name__ == "__main__":
    main()
