import argparse
from pathlib import Path
from datetime import datetime
import pygame
import os
import sys

from src.environment.game_env import SnakeGameEnv
from src.agents.dqn_agent import DQNAgent
from src.training.trainer import train, play
from src.utils.logger import setup_logger
from src.utils.config import GameParams, DQNParams

def main():
    """Main function to run the Snake RL training or play with a trained agent."""
    # Set up logger
    logger, timestamp = setup_logger()

    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

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
    parser.add_argument('--hidden_size', type=int, default=DQNParams.HIDDEN_SIZE,
                        help='Hidden layer size for neural network')
    parser.add_argument('--learning_rate', type=float, default=DQNParams.LEARNING_RATE,
                        help='Learning rate for the optimizer')
    parser.add_argument('--no_double_dqn', action='store_false', dest='double_dqn',
                        help='Disable Double DQN')
    parser.add_argument('--no_dueling_dqn', action='store_false', dest='dueling_dqn',
                        help='Disable Dueling DQN')
    parser.add_argument('--no_prioritized_replay', action='store_false', dest='prioritized_replay',
                        help='Disable Prioritized Experience Replay')

    # Set default values for the boolean arguments
    parser.set_defaults(double_dqn=DQNParams.USE_DOUBLE_DQN,
                       dueling_dqn=DQNParams.USE_DUELING_DQN,
                       prioritized_replay=DQNParams.USE_PRIORITIZED_REPLAY)

    args = parser.parse_args()

    # Print configuration
    logger.info(f"Running in {args.mode} mode")
    logger.info(f"Configuration: episodes={args.episodes}, render={args.render}, "
                f"render_freq={args.render_freq}, save_freq={args.save_freq}")
    logger.info(f"DQN settings: double_dqn={args.double_dqn}, dueling_dqn={args.dueling_dqn}, "
                f"prioritized_replay={args.prioritized_replay}, hidden_size={args.hidden_size}")

    # Set rendering mode based on arguments
    render_mode = "human" if args.mode == 'play' or args.render else "none"

    # Create environment
    env = SnakeGameEnv(render_mode=render_mode, fps=args.fps, snake_speed=args.speed)
    logger.info(f"Environment created with state_size={env.state_size}, action_size={env.action_size}")

    # Override DQN parameters with command line arguments
    DQNParams.USE_DOUBLE_DQN = args.double_dqn
    DQNParams.USE_DUELING_DQN = args.dueling_dqn
    DQNParams.USE_PRIORITIZED_REPLAY = args.prioritized_replay
    DQNParams.HIDDEN_SIZE = args.hidden_size
    DQNParams.LEARNING_RATE = args.learning_rate

    # Create agent
    agent = DQNAgent(env.state_size, env.action_size)

    # Load model if specified
    start_episode = 0
    if args.load:
        try:
            start_episode = agent.load_model(args.load)
            logger.info(f"Model loaded from {args.load}, starting from episode {start_episode}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            sys.exit(1)

    try:
        if args.mode == 'train':
            # Train the agent
            logger.info(f"Starting training for {args.episodes} episodes")
            train(agent, env, args.episodes, timestamp, save_freq=args.save_freq, render_freq=args.render_freq)
        elif args.mode == 'play':
            # Play with the trained agent
            logger.info("Starting play mode")
            play(agent, env)
    except KeyboardInterrupt:
        logger.info("Training/playing interrupted by user")
    except Exception as e:
        logger.error(f"Error during execution: {e}")
    finally:
        # Close environment
        env.close()
        logger.info("Environment closed")


if __name__ == "__main__":
    main()
