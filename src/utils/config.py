class Colors:
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    DARK_BLUE = (0, 0, 139)
    GRAY = (50, 50, 50)


class GameParams:
    # Grid dimensions
    GRID_WIDTH = 20
    GRID_HEIGHT = 20

    # Display dimensions
    CELL_SIZE = 20
    WIDTH = GRID_WIDTH * CELL_SIZE
    HEIGHT = GRID_HEIGHT * CELL_SIZE

    # Font size
    FONT_SIZE = 24

    # Game speed (higher value = slower snake)
    SNAKE_SPEED = 2  # Reduced from default for faster gameplay

    # FPS for rendering
    FPS = 60

    # Maximum steps per episode
    MAX_STEPS = 2000

    # Maximum steps without food before terminating
    MAX_STEPS_WITHOUT_FOOD = 100

    # State representation
    LOOK_AHEAD = 3  # How many cells ahead to look for dangers

    # Calculate state size based on parameters
    STATE_SIZE = 15 + (LOOK_AHEAD * 3)  # Base features + look ahead features


class DQNParams:
    # Core DQN parameters
    GAMMA = 0.99  # Discount factor
    LEARNING_RATE = 0.0005  # Higher learning rate for faster convergence
    EPSILON_START = 1.0  # Start with 100% exploration
    EPSILON_MIN = 0.01  # Minimum exploration rate
    EPSILON_DECAY = 0.995  # Faster decay for quicker convergence to exploitation

    # Memory parameters
    BATCH_SIZE = 64  # Larger batch size for more stable learning
    MEMORY_SIZE = 50000  # Replay memory size

    # Network update frequency
    TARGET_UPDATE_FREQ = 5  # Update target network more frequently

    # Advanced techniques
    USE_DOUBLE_DQN = True  # Use Double DQN to reduce overestimation
    USE_DUELING_DQN = True  # Use Dueling DQN for better value estimation
    USE_PRIORITIZED_REPLAY = True  # Use prioritized experience replay

    # Network architecture
    HIDDEN_SIZE = 128  # Size of hidden layers
    NUM_LAYERS = 2  # Number of hidden layers


class RewardParams:
    # Core rewards
    REWARD_FOOD = 10.0  # Reward for eating food
    REWARD_DEATH = -10.0  # Penalty for dying

    # Movement rewards
    REWARD_MOVE_TOWARDS_FOOD = 0.1  # Small reward for moving towards food
    REWARD_MOVE_AWAY_FROM_FOOD = -0.1  # Small penalty for moving away from food
    REWARD_SURVIVAL = 0.01  # Tiny reward for surviving each step

    # Additional rewards
    REWARD_FOOD_BONUS = 0.5  # Additional reward per snake length when eating food
    REWARD_STUCK_PENALTY = -5.0  # Penalty for getting stuck without finding food
