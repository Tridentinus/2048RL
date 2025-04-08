import pygame
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
from collections import deque
import time
import os
import matplotlib.pyplot as plt
from pynput.keyboard import Key, Controller
import argparse


# Constants from your original game
FPS = 60
WIDTH, HEIGHT = 800, 800
ROWS, COLS = 4, 4
RECT_HEIGHT = HEIGHT//ROWS
RECT_WIDTH = WIDTH//ROWS

# Set up keyboard controller for agent to control game
keyboard = Controller()

class DQNAgent:
    """Deep Q-Network agent for playing 2048 in pygame."""
    
    def __init__(self, state_shape=(4, 4), action_size=4):
        # Environment parameters
        self.state_shape = state_shape
        self.action_size = action_size
        
        # Learning parameters
        self.learning_rate = 0.0005
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        
        # Experience replay parameters
        self.memory = deque(maxlen=100000)
        self.batch_size = 64
        
        # Build models
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # Training stats
        self.train_episodes = 0
        self.best_score = 0
        self.scores = []
        self.max_tiles = []
        self.win_history = []
        
        # Action mapping (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
        self.key_mapping = {
            0: Key.up,
            1: Key.right,
            2: Key.down,
            3: Key.left
        }
        
    def _build_model(self):
        """Create a CNN model for Q-value approximation."""
        inputs = layers.Input(shape=self.state_shape)
        
        # Reshape to add channel dimension for CNN
        x = layers.Reshape((*self.state_shape, 1))(inputs)
        
        # CNN layers
        x = layers.Conv2D(64, (2, 2), padding='same', activation='relu')(x)
        x = layers.Conv2D(128, (2, 2), padding='same', activation='relu')(x)
        x = layers.Conv2D(128, (2, 2), padding='same', activation='relu')(x)
        
        # Flatten and dense layers
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)
        
        # Output layer (Q-values for each action)
        outputs = layers.Dense(self.action_size, activation='linear')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                     loss='mse')
        
        return model
    
    def update_target_model(self):
        """Copy weights from model to target_model."""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def get_state_from_tiles(self, tiles):
        """Convert tiles dictionary to a state representation."""
        # Create a 4x4 grid of zeros
        state = np.zeros((ROWS, COLS), dtype=float)
        
        # Fill in the values from the tiles
        for tile_key, tile in tiles.items():
            row, col = tile.row, tile.col
            # Use log2 representation
            state[row][col] = np.log2(tile.value) if tile.value > 0 else 0
            
        return state
    
    def get_valid_moves(self, tiles):
        """Determine which moves are valid in the current state."""
        # This is a simplified version - in real implementation you'd check
        # if the move would change the board state
        valid_moves = []
        
        # Check for each direction if there's a valid move
        # This is a simplified check - in practice you'd need to simulate each move
        board = self.get_state_from_tiles(tiles)
        
        # For now, let's assume all moves are valid unless blocked
        # UP (0)
        if not all(board[0, :] > 0):  # If top row has spaces
            valid_moves.append(0)
        # RIGHT (1)
        if not all(board[:, -1] > 0):  # If rightmost column has spaces
            valid_moves.append(1)
        # DOWN (2)
        if not all(board[-1, :] > 0):  # If bottom row has spaces
            valid_moves.append(2)
        # LEFT (3)
        if not all(board[:, 0] > 0):  # If leftmost column has spaces
            valid_moves.append(3)
            
        # Fallback: if no moves seem valid but game is not over, allow all moves
        if not valid_moves and len(tiles) < ROWS * COLS:
            valid_moves = [0, 1, 2, 3]
            
        return valid_moves
    
    def act(self, state, valid_moves=None):
        """Choose an action using epsilon-greedy policy."""
        state = np.expand_dims(state, axis=0)  # Add batch dimension
        
        if valid_moves is None or len(valid_moves) == 0:
            valid_moves = [0, 1, 2, 3]
        
        # Explore: choose a random valid action
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_moves)
        
        # Exploit: choose best valid action from Q-values
        act_values = self.model.predict(state, verbose=0)[0]
        
        # For visualization, return all Q-values and the chosen action
        # Filter for valid moves only and find best
        valid_q_values = [(action, act_values[action]) for action in valid_moves]
        best_action = max(valid_q_values, key=lambda x: x[1])[0]
        
        return best_action, act_values
    
    def press_key(self, action):
        """Simulate a keypress for the given action."""
        key = self.key_mapping[action]
        keyboard.press(key)
        time.sleep(0.1)  # Small delay to ensure key is registered
        keyboard.release(key)
    
    def replay(self):
        """Train the model using experience replay."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random minibatch
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch data
        states = np.zeros((self.batch_size, *self.state_shape))
        targets = np.zeros((self.batch_size, self.action_size))
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            
            # Compute target Q-value
            target = reward
            if not done:
                next_q_values = self.target_model.predict(np.expand_dims(next_state, axis=0), verbose=0)[0]
                target += self.gamma * np.max(next_q_values)
            
            # Get current Q-values and update the target for the chosen action
            targets[i] = self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0]
            targets[i][action] = target
        
        # Train the model
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        """Load model weights."""
        self.model.load_weights(name)
        self.update_target_model()
    
    def save(self, name):
        """Save model weights."""
        self.model.save_weights(name)


def draw_q_values(window, q_values, action, agent):
    """Draw Q-values and selected action for visualization."""
    # Colors for each action
    action_colors = [
        (0, 255, 0),    # UP - green
        (255, 165, 0),  # RIGHT - orange
        (255, 0, 0),    # DOWN - red
        (0, 0, 255)     # LEFT - blue
    ]
    
    # Create font
    font = pygame.font.SysFont("monospace", 24)
    
    # Labels for each action
    action_labels = ["UP", "RIGHT", "DOWN", "LEFT"]
    
    # Draw Q-values
    for i, (q, label) in enumerate(zip(q_values, action_labels)):
        color = action_colors[i]
        
        # Highlight selected action
        if i == action:
            # Draw rectangle background for selected action
            pygame.draw.rect(window, (255, 255, 200), 
                            (10, HEIGHT - 140 + i*30, 200, 30))
            # Make text bold
            font = pygame.font.SysFont("monospace", 24, bold=True)
        else:
            font = pygame.font.SysFont("monospace", 24)
            
        text = font.render(f"{label}: {q:.2f}", 1, color)
        window.blit(text, (20, HEIGHT - 130 + i*30))
    
    # Draw training stats
    font = pygame.font.SysFont("monospace", 20)
    text = font.render(f"ε: {agent.epsilon:.3f}", 1, (0, 0, 0))
    window.blit(text, (20, HEIGHT - 210))


def calculate_reward(prev_score, new_score, prev_max_tile, new_max_tile, prev_empty, new_empty, won):
    """Calculate reward for the agent based on game state changes."""
    reward = 0
    
    # Reward for score increase
    score_diff = new_score - prev_score
    reward += score_diff
    
    # Reward for increasing max tile
    if new_max_tile > prev_max_tile:
        # Logarithmic bonus for reaching new powers of 2
        reward += 10 * np.log2(new_max_tile / prev_max_tile)
    
    # Reward for maintaining empty spaces
    empty_diff = new_empty - prev_empty
    reward += empty_diff * 2
    
    # Big reward for winning
    if won:
        reward += 1000
    
    return reward


def agent_play_game(window, agent, training=True, fps=30, delay=0.1):
    """Let the agent play the game while visualizing the process."""
    from game2048 import Game, Tile, get_random_pos, draw, check_loss, end_move, move_tiles, update_tiles
    
    clock = pygame.time.Clock()
    run = True
    
    # Initialize the game
    game = Game()
    tiles = {}
    
    # Generate initial tiles
    for _ in range(2):
        row, col = get_random_pos(tiles)
        tiles[f"{row}{col}"] = Tile(2, row, col)
    
    # Track statistics
    episode_start_time = time.time()
    prev_score = 0
    prev_max_tile = 2
    move_count = 0
    
    # Main game loop
    while run:
        clock.tick(fps)
        
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:  # ESC to exit
                    run = False
                    break
                if event.key == pygame.K_SPACE:  # Space to pause/resume
                    # Wait for another space press
                    waiting = True
                    while waiting:
                        for pause_event in pygame.event.get():
                            if pause_event.type == pygame.KEYDOWN and pause_event.key == pygame.K_SPACE:
                                waiting = False
                            if pause_event.type == pygame.QUIT:
                                waiting = False
                                run = False
        
        # Get current state
        current_state = agent.get_state_from_tiles(tiles)
        
        # Count empty cells
        empty_cells = ROWS * COLS - len(tiles)
        
        # Get valid moves
        valid_moves = agent.get_valid_moves(tiles)
        
        # Let agent choose an action
        if training:
            action_data = agent.act(current_state, valid_moves)
            if isinstance(action_data, tuple):
                action, q_values = action_data
            else:
                action = action_data
                q_values = [0, 0, 0, 0]  # Placeholder if q_values not returned
        else:
            # When testing, always use best action (epsilon = 0)
            temp_epsilon = agent.epsilon
            agent.epsilon = 0
            action_data = agent.act(current_state, valid_moves)
            agent.epsilon = temp_epsilon
            if isinstance(action_data, tuple):
                action, q_values = action_data
            else:
                action = action_data
                q_values = [0, 0, 0, 0]
        
        # Map action to direction
        direction = ["up", "right", "down", "left"][action]
        
        # Execute the move
        move_result = move_tiles(window, tiles, clock, direction, game)
        game.update_score(move_result[1])
        move_count += 1
        
        # Check if game is over
        game_over = move_result[0] == "lost"
        
        # Check for win condition
        max_tile = max([tile.value for tile in tiles.values()]) if tiles else 0
        game_won = max_tile >= 2048
        
        # Calculate rewards
        current_score = game.score
        reward = calculate_reward(
            prev_score, current_score,
            prev_max_tile, max_tile,
            empty_cells, ROWS * COLS - len(tiles),
            game_won and prev_max_tile < 2048  # Only reward winning once
        )
        
        # Get new state
        new_state = agent.get_state_from_tiles(tiles)
        
        # Store experience for training
        if training:
            agent.remember(current_state, action, reward, new_state, game_over)
            
            # Train agent
            agent.replay()
            
            # Update target model occasionally
            if agent.train_episodes % 10 == 0:
                agent.update_target_model()
        
        # Update statistics
        prev_score = current_score
        prev_max_tile = max_tile
        
        # Visualize
        draw(window, tiles, game)
        draw_q_values(window, q_values, action, agent)
        
        # Draw additional stats
        font = pygame.font.SysFont("monospace", 20)
        text = font.render(f"Moves: {move_count}", 1, (0, 0, 0))
        window.blit(text, (20, HEIGHT - 240))
        
        text = font.render(f"Max Tile: {max_tile}", 1, (0, 0, 0))
        window.blit(text, (20, HEIGHT - 270))
        
        text = font.render(f"Mode: {'Training' if training else 'Testing'}", 1, (0, 0, 0))
        window.blit(text, (WIDTH - 200, HEIGHT - 270))
        
        pygame.display.update()
        
        # Add a small delay to better visualize agent's moves
        if delay > 0:
            time.sleep(delay)
        
        # Check for game over
        if game_over or game_won:
            # Record stats
            episode_duration = time.time() - episode_start_time
            agent.scores.append(current_score)
            agent.max_tiles.append(max_tile)
            agent.win_history.append(1 if game_won else 0)
            
            if training:
                agent.train_episodes += 1
                
                # Update best score
                if current_score > agent.best_score:
                    agent.best_score = current_score
                    # Save best model
                    agent.save("best_2048_model.weights.h5")
                
                # Save model periodically
                if agent.train_episodes % 50 == 0:
                    agent.save(f"2048_model_ep{agent.train_episodes}.h5")
                
                # Print info
                print(f"Episode {agent.train_episodes}: "
                      f"Score: {current_score}, Max Tile: {max_tile}, "
                      f"Moves: {move_count}, Duration: {episode_duration:.1f}s, "
                      f"Epsilon: {agent.epsilon:.4f}")
            
            # Display game over message
            font = pygame.font.SysFont("comicsans", 70, bold=True)
            if game_won:
                text = font.render("YOU WIN!", 1, (255, 50, 50))
            else:
                text = font.render("GAME OVER", 1, (255, 50, 50))
            
            text_rect = text.get_rect(center=(WIDTH/2, HEIGHT/2))
            pygame.draw.rect(window, (0, 0, 0, 128), text_rect.inflate(20, 20), border_radius=10)
            window.blit(text, text_rect)
            pygame.display.update()
            
            # Wait a bit before starting a new game
            time.sleep(2)
            
            # Reset the game
            game = Game()
            tiles.clear()
            for _ in range(2):
                row, col = get_random_pos(tiles)
                tiles[f"{row}{col}"] = Tile(2, row, col)
                
            move_count = 0
            prev_score = 0
            prev_max_tile = 2
            episode_start_time = time.time()
    
    # Save final model when closing
    if training:
        agent.save("2048_final_model.weights.h5")
    
    pygame.quit()
    return agent.scores, agent.max_tiles, agent.win_history


def plot_training_results(scores, max_tiles, win_history):
    """Plot training metrics."""
    plt.figure(figsize=(15, 5))
    
    # Plot scores
    plt.subplot(1, 3, 1)
    plt.plot(scores)
    plt.title('Game Score')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    # Plot max tiles
    plt.subplot(1, 3, 2)
    plt.plot(max_tiles)
    plt.title('Max Tile')
    plt.xlabel('Episode')
    plt.ylabel('Max Tile Value')
    
    # Plot win rate (moving average)
    plt.subplot(1, 3, 3)
    window_size = min(100, len(win_history))
    win_rate = [np.mean(win_history[max(0, i-window_size):i+1])*100 
                for i in range(len(win_history))]
    plt.plot(win_rate)
    plt.title(f'Win Rate ({window_size}-ep moving avg)')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate (%)')
    plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('2048_training_results.png')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='2048 Game with DQN Agent')
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'test', 'manual'],
                        help='Mode to run the agent in (train, test, or manual)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model file to load (default: None)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second (default: 30)')
    parser.add_argument('--delay', type=float, default=0.1,
                        help='Delay between agent moves (default: 0.1)')
    
    args = parser.parse_args()
    
    # Initialize pygame
    pygame.init()
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("2048 - DQN Agent")
    
    # Create agent
    agent = DQNAgent(state_shape=(4, 4), action_size=4)
    
    # Load model if specified
    if args.model and os.path.exists(args.model):
        print(f"Loading model from {args.model}")
        agent.load(args.model)
        # Set lower epsilon for loaded models
        if args.mode == 'train':
            agent.epsilon = 0.1
    
    if args.mode == 'manual':
        # Let human play the game
        from game2048 import main as play_game
        play_game(window)
    elif args.mode == 'train':
        # Train the agent
        scores, max_tiles, win_history = agent_play_game(
            window, agent, training=True, fps=args.fps, delay=args.delay)
        
        # Plot results
        plot_training_results(scores, max_tiles, win_history)
    elif args.mode == 'test':
        # Test the agent
        scores, max_tiles, win_history = agent_play_game(
            window, agent, training=False, fps=args.fps, delay=args.delay)
        
        # Print results
        print(f"Test Results ({len(scores)} episodes):")
        print(f"Avg Score: {np.mean(scores):.1f}")
        print(f"Avg Max Tile: {np.mean(max_tiles):.1f}")
        print(f"Win Rate: {np.mean(win_history)*100:.1f}%")


if __name__ == "__main__":
    main()
