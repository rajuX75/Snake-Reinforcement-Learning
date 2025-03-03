import time
import pygame
import logging
from pathlib import Path

logger = logging.getLogger("snake_rl")

def train(agent, env, num_episodes, timestamp, save_freq=100, render_freq=20):
    """
    Train the DQN agent on the Snake game.

    Args:
        agent: The DQN agent
        env: The game environment
        num_episodes: Number of training episodes
        timestamp: Timestamp for saving files
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
                        agent.plot_stats(timestamp, show=False, save=True)
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
            agent.save_model(episode, score, timestamp)

        # Save best model
        if score > best_score:
            best_score = score
            agent.save_model(episode, score, timestamp)
            logger.info(f"New best score: {best_score} (Episode {episode})")

    # Final save
    agent.save_model(num_episodes, score, timestamp)

    # Plot statistics
    agent.plot_stats(timestamp, show=False, save=True)

    logger.info(f"Training completed. Best score: {best_score}")

    return agent


def play(agent, env, num_episodes=100):
    """
    Play the Snake game with a trained agent.

    Args:
        agent: The trained DQN agent
        env: The game environment
        num_episodes: Number of episodes to play
    """
    for episode in range(1, num_episodes + 1):
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
