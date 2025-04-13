import numpy as np
import random
import math
import time
from game2048 import timer

winTile = 64
def timer(func):
    """Decorator to time function execution"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        # print(f"{func.__name__} took {end_time - start_time:.6f} seconds")
        return result
    return wrapper
class Game2048Sim:
    """
    A headless simulation environment for 2048 that mirrors the game logic
    but without any pygame or visualization overhead.
    """
    
    def __init__(self):
        self.rows = 3
        self.cols = 3
        self.reset()
    
    def reset(self):
        """Reset the game to initial state."""
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.score = 0
        self.high_score = 0
        self.done = False
        self.won = False
        
        # Add two initial tiles
        self.add_random_tile()
        self.add_random_tile()
        
        return self.get_state()
    
    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell."""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            row, col = random.choice(empty_cells)
            self.board[row, col] = 2 if random.random() < 0.9 else 4
    
    def get_state(self):
        """Get the current state in log2 form for the agent."""
        log_board = np.zeros_like(self.board, dtype=float)
        mask = self.board > 0
        log_board[mask] = np.log2(self.board[mask])
        return log_board
    # @timer
    def get_valid_moves(self):
        """Get valid moves (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)."""
        valid_moves = []
        directions = ["up", "right", "down", "left"]
        
        for action, direction in enumerate(directions):
            if self._is_valid_move(direction):
                valid_moves.append(action)
                
        return valid_moves
    # @timer
    def _is_valid_move(self, direction):
        """Check if a move in the given direction would change the board state."""
        # Create a copy of the board to simulate the move
        board_copy = self.board.copy()
        merged, _ = self._move(board_copy, direction)
        return merged
    # @timer
    def _move(self, board, direction):
        """
        Execute a move on the provided board.
        Returns (moved, score_gain)
        """
        moved = False
        score_gain = 0
        
        # Handle different directions by rotating the board
        if direction == "up":
            board = np.rot90(board, k=1)
        elif direction == "right":
            board = np.rot90(board, k=2)
        elif direction == "down":
            board = np.rot90(board, k=3)
        
        # Process each row (left to right)
        for row in range(self.rows):
            # Remove zeros and slide tiles to the left
            row_data = board[row][board[row] > 0]
            
            # Skip empty rows
            if len(row_data) == 0:
                continue
                
            # Create a new row with the tiles moved
            new_row = np.zeros(self.cols, dtype=int)
            write_idx = 0
            read_idx = 0
            
            while read_idx < len(row_data):
                # If this is the last tile or next tile is different, just place it
                if read_idx == len(row_data) - 1 or row_data[read_idx] != row_data[read_idx + 1]:
                    new_row[write_idx] = row_data[read_idx]
                    read_idx += 1
                else:
                    # Merge two equal tiles
                    merged_value = row_data[read_idx] * 2
                    new_row[write_idx] = merged_value
                    score_gain += merged_value
                    read_idx += 2  # Skip the merged tiles
                    
                write_idx += 1
            
            # Check if the row changed
            if not np.array_equal(board[row], new_row):
                moved = True
                
            board[row] = new_row
        
        # Rotate the board back to its original orientation
        if direction == "up":
            board = np.rot90(board, k=3)
        elif direction == "right":
            board = np.rot90(board, k=2)
        elif direction == "down":
            board = np.rot90(board, k=1)
        
        return moved, score_gain
    # @timer
    def step(self, action):
        """
        Take an action and update the game state.
        
        Parameters:
        - action: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        
        Returns:
        - next_state: The new game state
        - reward: The reward for this step
        - done: Whether the game is over
        - info: Additional information dictionary
        """
        if self.done:
            return self.get_state(), 0, True, {"score": self.score, "won": self.won}
        
        # Convert action number to direction string
        directions = ["up", "right", "down", "left"]
        direction = directions[action]
        
        # Check if move is valid
        if not self._is_valid_move(direction):
            return self.get_state(), -10, self.done, {
                "score": self.score, 
                "won": self.won,
                "invalid_move": True
            }
        
        # Get current state for reward calculation
        prev_empty = np.sum(self.board == 0)
        prev_max_tile = np.max(self.board)
        
        # Apply the move
        _, score_gain = self._move(self.board, direction)
        
        # Update game score
        self.score += score_gain
        if self.score > self.high_score:
            self.high_score = self.score
        
        # Add a new random tile
        self.add_random_tile()
        
        # Check for win condition (2048 tile)
        current_max = np.max(self.board)
        if current_max >= winTile and not self.won:
            self.won = True
        
        # Check for game over (no valid moves)
        if not self.get_valid_moves():
            self.done = True
        
        # Calculate reward
        reward = self._calculate_reward(
            score_gain=score_gain,
            prev_max_tile=prev_max_tile,
            current_max=current_max,
            prev_empty=prev_empty,
            current_empty=np.sum(self.board == 0),
            newly_won=current_max >= winTile and prev_max_tile < winTile
        )
        
        return self.get_state(), reward, self.done, {
            "score": self.score,
            "won": self.won,
            "max_tile": current_max,
            "reward": reward
        }
    
    def _calculate_reward(self, score_gain, prev_max_tile, current_max, 
                         prev_empty, current_empty, newly_won):
        """Calculate the reward for the current step."""
        reward = 0
        
        # Base reward from score gain
        # print(f"Score gain: {score_gain}")
        reward += score_gain
        
        
        # Reward for increasing max tile
        if current_max > prev_max_tile:
            # Logarithmic bonus for reaching new powers of 2
            progress_reward = 20 * (np.log2(current_max/prev_max_tile )**1.5)
            # print(f"Max tile reward (reached {current_max}): {progress_reward}")
            reward += progress_reward
        
        # Reward/penalty for empty tiles
        # empty_diff = current_empty - prev_empty
        # reward += empty_diff * 2
        
        # Big reward for winning
        if newly_won:
            # print("You won!, reward: 1000")
            reward += 1000
        
        # Extra penalties for game over without winning
        # if self.done and not self.won:
        #     reward -= 100
        
        return reward
    
    def render_text(self):
        """Render the game as text for debugging."""
        print(f"Score: {self.score}  High Score: {self.high_score}")
        
        # Format each cell
        print("+" + "-----+" * self.cols)
        for row in range(self.rows):
            row_str = "|"
            for col in range(self.cols):
                value = self.board[row, col]
                if value == 0:
                    row_str += "     |"
                else:
                    # Right-align the number in a 5-character field
                    row_str += f"{value:5d}|"
            print(row_str)
            print("+" + "-----+" * self.cols)
        
        if self.won:
            print("YOU WON!")
        if self.done and not self.won:
            print("GAME OVER!")


def interactive_play():
    """Allow the user to play the game via command line."""
    game = Game2048Sim()
    
    # Action mapping
    action_map = {
        'w': 0,  # UP
        'd': 1,  # RIGHT
        's': 2,  # DOWN
        'a': 3,  # LEFT
    }
    
    print("\n===== 2048 SIMULATION =====")
    print("Controls: w (UP), d (RIGHT), s (DOWN), a (LEFT), q (QUIT)")
    print("=========================\n")
    
    game.render_text()
    total_reward = 0

    while not game.done:
        # Show valid moves
        valid_moves = game.get_valid_moves()
        valid_directions = [["UP", "RIGHT", "DOWN", "LEFT"][m] for m in valid_moves]
        print(f"Valid moves: {', '.join(valid_directions)}")
        # Get user input
        user_input = input("Enter move (w/d/s/a) or q to quit: ").lower()
        
        if user_input == 'q':
            print("Quitting game...")
            break
            
        if user_input not in action_map:
            print("Invalid input! Use w/d/s/a for movement or q to quit.")
            continue
            
        action = action_map[user_input]
        
        # Check if move is valid
        if action not in valid_moves:
            print(f"Invalid move! That direction won't change the board.")
            continue

        # Execute the move
        _, reward, done, info = game.step(action)
        total_reward += reward
        # Clear the screen (optional, may not work in all environments)
        print("\n" * 5)
        
        # Show the game state
        game.render_text()
        print(f"Reward: {reward}")
        print(f"Total Reward: {total_reward}")
        if done:
            if game.won:
                print("Congratulations! You won!")
            else:
                print("Game over! No more moves available.")
                
    print(f"Final score: {game.score}")
    print(f"Max tile: {np.max(game.board)}")


def agent_play(agent, visual_delay=0.5, max_steps=1000):
    """Have an agent play the game with visualization."""
    game = Game2048Sim()
    state = game.get_state()
    done = False
    total_reward = 0
    steps = 0
    
    print("\n===== 2048 AGENT SIMULATION =====")
    game.render_text()
    
    while not done and steps < max_steps:
        # Get valid moves
        valid_moves = game.get_valid_moves()
        
        # Agent chooses action
        action_data = agent.act(state, valid_moves)
        action = action_data[0] if isinstance(action_data, tuple) else action_data
        
        # Show agent's decision
        direction = ["UP", "RIGHT", "DOWN", "LEFT"][action]
        print(f"Agent chooses: {direction}")
        
        # Execute the move
        next_state, reward, done, info = game.step(action)
        total_reward += reward
        # print(f"Total Reward: {reward}")
        steps += 1
        
        # Update state
        state = next_state
        
        # Delay for visualization
        time.sleep(visual_delay)
        
        # Clear screen and show new state
        print("\n" * 5)
        game.render_text()
        print(f"Step {steps}, Total Reward: {total_reward}")
        
        if done:
            if game.won:
                print("Agent won the game!")
            else:
                print("Game over! No more moves available.")
    
    print(f"Game finished after {steps} steps")
    print(f"Final score: {game.score}")
    print(f"Max tile: {info.get('max_tile', np.max(game.board))}")
    print(f"Total reward: {total_reward}")
    
    return game.score, info.get('max_tile', np.max(game.board)), game.won


def main():
    """Main function to run the game with user interaction."""
    print("Welcome to 2048 Simulation!")
    print("1. Play yourself")
    print("2. Watch trained agent play")
    print("3. Train agent (fast)")
    print("4. Exit")
    
    choice = input("Enter your choice (1-4): ")
    
    if choice == "1":
        interactive_play()
    elif choice == "2":
        # Check if agent is available
        try:
            from dqn_agent import DQNAgent
            agent = DQNAgent()
            
            # Ask for model file
            model_file = input("Enter path to model file (or press Enter for random agent): ")
            if model_file.strip():
                try:
                    agent.load(model_file)
                    print(f"Loaded model from {model_file}")
                except:
                    print("Couldn't load model. Using random agent.")
                    agent.epsilon = 1.0  # Force exploration (random moves)
            else:
                print("Using random agent (no model loaded).")
                agent.epsilon = 1.0
                
            # Ask for visualization speed
            try:
                delay = float(input("Enter delay between moves in seconds (default: 0.5): ") or 0.5)
            except:
                delay = 0.5
                
            agent_play(agent, visual_delay=delay)
        except ImportError:
            print("Agent implementation not available. Please run 'python training_manager.py' instead.")
    elif choice == "3":
        print("For training, please use 'python training_manager.py --mode sim_train'")
    elif choice == "4":
        print("Goodbye!")
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
