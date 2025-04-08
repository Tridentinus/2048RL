import argparse
import time
import os
from dqn_agent import DQNAgent
import pygame
from pynput.keyboard import Key, Controller

def import_pygame_game():
    """Import the pygame implementation of 2048."""
    try:
        # Try import your original game
        from game2048 import Game, Tile, get_random_pos, draw, WINDOW
        return True
    except ImportError:
        print("Original pygame implementation not found.")
        return False

def train_with_pygame(agent, fps=30, delay=0.1, episodes=None, model_path=None):
    """Train the agent using the pygame visualization."""
    pygame_available = import_pygame_game()
    if not pygame_available:
        print("Cannot train with pygame visualization. Original game not found.")
        return
    
    # Import the pygame training function
    from pygame_agent import agent_play_game
    
    # Initialize pygame
    pygame.init()
    window = pygame.display.set_mode((800, 800))
    pygame.display.set_caption("2048 - DQN Agent Training")
    
    # Load model if specified
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        agent.load(model_path)
        # Set lower epsilon for continued training
        agent.epsilon = 0.1
    
    # Train the agent
    scores, max_tiles, win_history = agent_play_game(
        window, agent, training=True, fps=fps, delay=delay, max_episodes=episodes)
    
    pygame.quit()
    
    # Plot training results
    agent.plot_training_results()
    
    return scores, max_tiles, win_history

def train_with_simulation(agent, episodes=10000, model_path=None):
    """Train the agent using the fast simulation."""
    from sim2048 import Game2048Sim
    
    # Create environment
    env = Game2048Sim()
    
    # Load model if specified
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        agent.load(model_path)
        # Set lower epsilon for continued training
        agent.epsilon = 0.1
    
    print(f"Starting fast simulation training for {episodes} episodes...")
    start_time = time.time()
    
    # Train agent
    scores, max_tiles, win_history = agent.train_headless(
        env, episodes=episodes, save_every=500, print_every=100)
    
    # Calculate training speed
    end_time = time.time()
    training_time = end_time - start_time
    eps_per_second = episodes / training_time
    
    print(f"Training complete! {episodes} episodes in {training_time:.1f} seconds")
    print(f"Training speed: {eps_per_second:.2f} episodes per second")
    
    # Plot training results
    agent.plot_training_results()
    
    return scores, max_tiles, win_history

def test_with_pygame(agent, model_path, fps=10, delay=0.3, episodes=10):
    """Test the agent using the pygame visualization."""
    pygame_available = import_pygame_game()
    if not pygame_available:
        print("Cannot test with pygame visualization. Original game not found.")
        return
    
    # Import the pygame testing function
    from pygame_agent import agent_play_game
    
    # Initialize pygame
    pygame.init()
    window = pygame.display.set_mode((800, 800))
    pygame.display.set_caption("2048 - DQN Agent Testing")
    
    # Load model
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return
    
    print(f"Loading model from {model_path}")
    agent.load(model_path)
    
    # Test the agent
    scores, max_tiles, win_history = agent_play_game(
        window, agent, training=False, fps=fps, delay=delay, max_episodes=episodes)
    
    pygame.quit()
    
    return scores, max_tiles, win_history

def test_with_simulation(agent, model_path, episodes=100):
    """Test the agent using the fast simulation."""
    from game2048_sim import Game2048Sim
    
    # Create environment
    env = Game2048Sim()
    
    # Load model
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return
    
    print(f"Loading model from {model_path}")
    agent.load(model_path)
    
    # Test agent
    scores, max_tiles, win_rate = agent.test_headless(env, episodes=episodes)
    
    return scores, max_tiles, win_rate

def play_manually():
    """Let the human play the game manually."""
    pygame_available = import_pygame_game()
    if not pygame_available:
        print("Cannot play manually. Original game not found.")
        return
    
    from game2048 import main as play_game
    
    # Initialize pygame
    pygame.init()
    window = pygame.display.set_mode((800, 800))
    pygame.display.set_caption("2048 - Human Play")
    
    # Start the game
    play_game(window)
    
    pygame.quit()

def main():
    parser = argparse.ArgumentParser(description='2048 Game Training Manager')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='sim_train', 
                        choices=['sim_train', 'vis_train', 'sim_test', 'vis_test', 'manual'],
                        help='Mode to run: sim_train (simulation training), vis_train (visual training), '
                             'sim_test (simulation testing), vis_test (visual testing), manual (human play)')
    
    # Model file options
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model file to load (default: None)')
    
    # Training options
    parser.add_argument('--episodes', type=int, default=10000,
                        help='Number of episodes for training/testing (default: 10000 for training, 100 for testing)')
    
    # Visualization options
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for visualization (default: 30)')
    parser.add_argument('--delay', type=float, default=0.1,
                        help='Delay between agent moves in visualization (default: 0.1)')
    
    args = parser.parse_args()
    
    # Create agent
    agent = DQNAgent(state_shape=(4, 4), action_size=4)
    
    if args.mode == 'sim_train':
        # Fast simulation training
        episodes = args.episodes if args.episodes else 10000
        train_with_simulation(agent, episodes=episodes, model_path=args.model)
        
    elif args.mode == 'vis_train':
        # Visual training with pygame
        episodes = args.episodes if args.episodes else 1000
        train_with_pygame(agent, fps=args.fps, delay=args.delay, episodes=episodes, model_path=args.model)
        
    elif args.mode == 'sim_test':
        # Fast simulation testing
        episodes = args.episodes if args.episodes else 100
        test_with_simulation(agent, model_path=args.model, episodes=episodes)
        
    elif args.mode == 'vis_test':
        # Visual testing with pygame
        episodes = args.episodes if args.episodes else 10
        test_with_pygame(agent, model_path=args.model, fps=args.fps, delay=args.delay, episodes=episodes)
        
    elif args.mode == 'manual':
        # Human play
        play_manually()
    
    else:
        print("Invalid mode selected.")

if __name__ == "__main__":
    main()
