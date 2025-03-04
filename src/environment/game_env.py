import pygame
import numpy as np
import torch
import random
import logging
from collections import deque
from enum import Enum

from src.utils.config import GameParams

logger = logging.getLogger("snake_rl")

class Direction(Enum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

class SnakeGameEnv:
    def __init__(self, render_mode="none", fps=GameParams.FPS, snake_speed=GameParams.SNAKE_SPEED):
        self.render_mode = render_mode
        self.fps = fps
        self.snake_speed = snake_speed
        self.width = GameParams.WIDTH
        self.height = GameParams.HEIGHT
        self.block_size = GameParams.BLOCK_SIZE
        self.max_steps_without_food = GameParams.MAX_STEPS_WITHOUT_FOOD

        # Game state
        self.snake = None
        self.direction = None
        self.food = None
        self.score = 0
        self.steps_without_food = 0
        self.game_over = False

        # Initialize pygame if rendering is enabled
        if self.render_mode == "human":
            pygame.init()
            self.font = pygame.font.SysFont('arial', 25)
            self.clock = pygame.time.Clock()
            self.display = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('Snake RL')

        # Define action and state spaces
        self.action_size = 4  # 0: right, 1: down, 2: left, 3: up

        # Rich state representation
        self.vision_range = 5  # How far the snake can "see"
        self.state_size = 11 + 4  # 11 features + one-hot encoding of direction

        logger.info(f"SnakeGameEnv initialized with render_mode={render_mode}, fps={fps}, snake_speed={snake_speed}")
        logger.info(f"State size: {self.state_size}, Action size: {self.action_size}")

    def reset(self):
        # Initialize snake
        x = self.width // 2
        y = self.height // 2
        self.snake = deque([(x, y)])
        self.direction = Direction.RIGHT

        # Initialize food
        self.food = self._place_food()

        # Reset game state
        self.score = 0
        self.steps_without_food = 0
        self.game_over = False

        # Return initial state
        return self._get_state()

    def step(self, action):
        # Map action to direction
        # 0: right, 1: down, 2: left, 3: up
        new_direction = Direction(action)

        # Prevent 180-degree turns
        if (new_direction.value + 2) % 4 != self.direction.value:
            self.direction = new_direction

        # Move snake
        head_x, head_y = self.snake[0]
        if self.direction == Direction.RIGHT:
            head_x += self.block_size
        elif self.direction == Direction.LEFT:
            head_x -= self.block_size
        elif self.direction == Direction.DOWN:
            head_y += self.block_size
        elif self.direction == Direction.UP:
            head_y -= self.block_size

        # Check if snake hits the wall
        if (head_x < 0 or head_x >= self.width or
            head_y < 0 or head_y >= self.height):
            self.game_over = True
            return self._get_state(), self._get_reward(collision=True), True, self.score

        # Check if snake hits itself
        if (head_x, head_y) in list(self.snake)[1:]:
            self.game_over = True
            return self._get_state(), self._get_reward(collision=True), True, self.score

        # Move snake
        self.snake.appendleft((head_x, head_y))

        # Check if snake eats food
        reward = 0
        if (head_x, head_y) == self.food:
            self.score += 1
            reward = self._get_reward(ate_food=True)
            self.food = self._place_food()
            self.steps_without_food = 0
        else:
            self.snake.pop()
            reward = self._get_reward()
            self.steps_without_food += 1

        # Check if snake is stuck
        if self.steps_without_food >= self.max_steps_without_food:
            self.game_over = True
            return self._get_state(), self._get_reward(stuck=True), True, self.score

        return self._get_state(), reward, self.game_over, self.score

    def _get_reward(self, ate_food=False, collision=False, stuck=False):
        if collision:
            return GameParams.COLLISION_REWARD
        elif stuck:
            return GameParams.STUCK_REWARD
        elif ate_food:
            return GameParams.FOOD_REWARD
        else:
            # Calculate distance-based reward
            head_x, head_y = self.snake[0]
            food_x, food_y = self.food

            # Manhattan distance to food
            curr_dist = abs(head_x - food_x) + abs(head_y - food_y)

            # Encourage moving towards food
            return GameParams.STEP_REWARD

    def _place_food(self):
        while True:
            food_x = random.randint(0, (self.width - self.block_size) // self.block_size) * self.block_size
            food_y = random.randint(0, (self.height - self.block_size) // self.block_size) * self.block_size

            # Make sure food doesn't spawn on snake
            if (food_x, food_y) not in self.snake:
                return (food_x, food_y)

    def _get_state(self):
        """
        Rich state representation with:
        1. Danger detection in all directions
        2. Relative food position (normalized)
        3. Current direction (one-hot encoded)
        4. Body awareness
        """
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food

        # Initialize state vector
        state = []

        # 1. Look-ahead danger detection (4 values)
        # Check for danger straight ahead, right, and left relative to current direction
        danger_straight = False
        danger_right = False
        danger_left = False
        danger_back = False

        # Calculate positions for danger detection based on current direction
        if self.direction == Direction.RIGHT:
            # Straight
            pos_straight = (head_x + self.block_size, head_y)
            # Right (from perspective of snake)
            pos_right = (head_x, head_y + self.block_size)
            # Left (from perspective of snake)
            pos_left = (head_x, head_y - self.block_size)
            # Back
            pos_back = (head_x - self.block_size, head_y)
        elif self.direction == Direction.LEFT:
            pos_straight = (head_x - self.block_size, head_y)
            pos_right = (head_x, head_y - self.block_size)
            pos_left = (head_x, head_y + self.block_size)
            pos_back = (head_x + self.block_size, head_y)
        elif self.direction == Direction.DOWN:
            pos_straight = (head_x, head_y + self.block_size)
            pos_right = (head_x - self.block_size, head_y)
            pos_left = (head_x + self.block_size, head_y)
            pos_back = (head_x, head_y - self.block_size)
        elif self.direction == Direction.UP:
            pos_straight = (head_x, head_y - self.block_size)
            pos_right = (head_x + self.block_size, head_y)
            pos_left = (head_x - self.block_size, head_y)
            pos_back = (head_x, head_y + self.block_size)

        # Check if these positions are dangerous (wall or body)
        danger_straight = (
            pos_straight[0] < 0 or
            pos_straight[0] >= self.width or
            pos_straight[1] < 0 or
            pos_straight[1] >= self.height or
            pos_straight in list(self.snake)[1:]
        )

        danger_right = (
            pos_right[0] < 0 or
            pos_right[0] >= self.width or
            pos_right[1] < 0 or
            pos_right[1] >= self.height or
            pos_right in list(self.snake)[1:]
        )

        danger_left = (
            pos_left[0] < 0 or
            pos_left[0] >= self.width or
            pos_left[1] < 0 or
            pos_left[1] >= self.height or
            pos_left in list(self.snake)[1:]
        )

        danger_back = (
            pos_back[0] < 0 or
            pos_back[0] >= self.width or
            pos_back[1] < 0 or
            pos_back[1] >= self.height or
            pos_back in list(self.snake)[1:] or
            len(self.snake) > 1  # Back is always dangerous if snake length > 1
        )

        state.append(float(danger_straight))
        state.append(float(danger_right))
        state.append(float(danger_left))
        state.append(float(danger_back))

        # 2. Relative food position (normalized)
        # Normalize by dividing by the maximum possible distance
        max_distance = self.width + self.height

        # Food direction relative to head
        food_right = float(food_x > head_x)
        food_left = float(food_x < head_x)
        food_down = float(food_y > head_y)
        food_up = float(food_y < head_y)

        state.append(food_right)
        state.append(food_left)
        state.append(food_down)
        state.append(food_up)

        # Normalized distance to food
        food_distance_x = abs(head_x - food_x) / self.width
        food_distance_y = abs(head_y - food_y) / self.height
        state.append(food_distance_x)
        state.append(food_distance_y)

        # 3. Current direction (one-hot encoded)
        dir_right = float(self.direction == Direction.RIGHT)
        dir_down = float(self.direction == Direction.DOWN)
        dir_left = float(self.direction == Direction.LEFT)
        dir_up = float(self.direction == Direction.UP)

        state.append(dir_right)
        state.append(dir_down)
        state.append(dir_left)
        state.append(dir_up)

        # 4. Body awareness - length of snake (normalized)
        state.append(len(self.snake) / (self.width * self.height / (self.block_size * self.block_size)))

        return torch.tensor([state], dtype=torch.float32)

    def render(self):
        if self.render_mode != "human":
            return

        self.display.fill((0, 0, 0))

        # Draw snake
        for i, (x, y) in enumerate(self.snake):
            if i == 0:  # Head
                pygame.draw.rect(self.display, (0, 255, 0), [x, y, self.block_size, self.block_size])
            else:  # Body
                pygame.draw.rect(self.display, (0, 200, 0), [x, y, self.block_size, self.block_size])
                pygame.draw.rect(self.display, (0, 255, 0), [x, y, self.block_size, self.block_size], 1)

        # Draw food
        pygame.draw.rect(self.display, (255, 0, 0), [self.food[0], self.food[1], self.block_size, self.block_size])

        # Draw score
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.display.blit(score_text, [0, 0])

        pygame.display.update()
        self.clock.tick(self.fps // self.snake_speed)

    def close(self):
        if self.render_mode == "human":
            pygame.quit()
