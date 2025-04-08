import numpy as np
import tensorflow as tf
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
from tensorflow import keras
from tensorflow.keras import layers

import random
from collections import deque
import os
import matplotlib.pyplot as plt
import time
def timer(func):
    """Decorator to time function execution"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        # print(f"{func.__name__} took {end_time - start_time:.6f} seconds")
        return result
    return wrapper
class DQNAgent:
    """
    Environment-agnostic DQN agent for playing 2048.
    Can work with both visualization-based and simulation-based environments.
    """
    
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
        self.batch_size = 32
        
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
    @timer
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
    @timer
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
        
        # Filter for valid moves only and find best
        valid_q_values = [(action, act_values[action]) for action in valid_moves]
        best_action = max(valid_q_values, key=lambda x: x[1])[0]
        
        # For visualization purposes, return action and all Q-values
        return best_action, act_values
    @timer
    def replay(self):
      if len(self.memory) < self.batch_size:
          return
      
      # Sample batch
      minibatch = random.sample(self.memory, self.batch_size)
      
      # Prepare batch data
      states = np.array([experience[0] for experience in minibatch])
      actions = np.array([experience[1] for experience in minibatch])
      rewards = np.array([experience[2] for experience in minibatch])
      next_states = np.array([experience[3] for experience in minibatch])
      dones = np.array([experience[4] for experience in minibatch])
      
      # Get all predictions in one batch
      current_q_values = self.model.predict(states, verbose=0)
      next_q_values = self.target_model.predict(next_states, verbose=0)
      
      # Calculate targets
      targets = current_q_values.copy()
      for i in range(self.batch_size):
          if dones[i]:
              targets[i, actions[i]] = rewards[i]
          else:
              targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
      
      # Train model
      self.model.fit(states, targets, epochs=1, verbose=0, batch_size=self.batch_size)
    
    def load(self, name):
        """Load model weights."""
        self.model.load_weights(name)
        self.update_target_model()
    
    def save(self, name):
        """Save model weights."""
        self.model.save_weights(name)
    
    def train_headless(self, env, episodes=10000, save_every=500, print_every=100, checkpoint_dir="checkpoints"):
        """
        Train the agent on a headless environment for fast training.
        
        Parameters:
        - env: The game environment (must implement reset(), step(), get_valid_moves())
        - episodes: Number of episodes to train for
        - save_every: Save the model every X episodes
        - print_every: Print progress every X episodes
        - checkpoint_dir: Directory to save checkpoints
        
        Returns:
        - Training statistics (scores, max_tiles, win_history)
        """
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        start_time = time.time()
        
        for episode in range(episodes):
            # Reset the environment
            print(episode)
            state = env.reset()
            total_reward = 0
            done = False
            
            # Track episode statistics
            steps = 0
            max_tile = 0
            
            while not done:
                # Get valid moves
                valid_moves = env.get_valid_moves()
                # print(valid_moves)
                # Choose action
                action_info = self.act(state, valid_moves)
                action = action_info[0] if isinstance(action_info, tuple) else action_info
                
                # Take action
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                
                # Update maximum tile
                if 'max_tile' in info:
                    max_tile = max(max_tile, info['max_tile'])
                
                # Remember experience
                self.remember(state, action, reward, next_state, done)
                
                # Move to next state
                state = next_state
                
                # Learn from experiences
                # self.replay()
                if steps % 4 == 0:
                  self.replay()
                # Update target model periodically
                if steps % 500 == 0:
                    self.update_target_model()
                    
                steps += 1
            
            # Record episode results
            self.train_episodes += 1
            self.scores.append(info['score'])
            self.max_tiles.append(max_tile)
            self.win_history.append(1 if info['won'] else 0)
            
            # Update best score
            if info['score'] > self.best_score:
                self.best_score = info['score']
                self.save(os.path.join(checkpoint_dir, "best_model.weights.h5"))
            
            # Print progress
            if episode % print_every == 0 or episode == episodes - 1:
                avg_score = np.mean(self.scores[-print_every:]) if self.scores else 0
                avg_max_tile = np.mean(self.max_tiles[-print_every:]) if self.max_tiles else 0
                win_rate = np.mean(self.win_history[-print_every:]) * 100 if self.win_history else 0
                elapsed = time.time() - start_time
                
                print(f"Episode: {episode}/{episodes} | "
                      f"Score: {info['score']} | "
                      f"Max Tile: {max_tile} | "
                      f"Steps: {steps} | "
                      f"Avg Score: {avg_score:.1f} | "
                      f"Win Rate: {win_rate:.1f}% | "
                      f"Epsilon: {self.epsilon:.4f} | "
                      f"Elapsed: {elapsed:.1f}s")
            
            # Save model periodically
            if episode > 0 and episode % save_every == 0:
                self.save(os.path.join(checkpoint_dir, f"model_ep{episode}.h5"))
        
        # Final save
        self.save(os.path.join(checkpoint_dir, "final_model.h5"))
        
        # Calculate training duration
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.1f} seconds.")
        print(f"Final win rate: {np.mean(self.win_history[-1000:]) * 100:.1f}%")
        
        return self.scores, self.max_tiles, self.win_history
    
    def test_headless(self, env, episodes=100):
        """
        Test the agent on a headless environment.
        
        Parameters:
        - env: The game environment 
        - episodes: Number of episodes to test for
        
        Returns:
        - Test statistics (scores, max_tiles, win_rate)
        """
        scores = []
        max_tiles = []
        wins = 0
        
        # Store original epsilon and set to minimum for testing
        original_epsilon = self.epsilon
        self.epsilon = self.epsilon_min
        
        for episode in range(episodes):
            state = env.reset()
            done = False
            
            while not done:
                valid_moves = env.get_valid_moves()
                action_info = self.act(state, valid_moves)
                action = action_info[0] if isinstance(action_info, tuple) else action_info
                
                next_state, _, done, info = env.step(action)
                state = next_state
            
            scores.append(info['score'])
            max_tiles.append(info.get('max_tile', 0))
            if info.get('won', False):
                wins += 1
            
            if episode % 10 == 0:
                print(f"Test Episode {episode}/{episodes} | "
                      f"Score: {info['score']} | "
                      f"Max Tile: {info.get('max_tile', 0)}")
        
        # Restore original epsilon
        self.epsilon = original_epsilon
        
        # Print test results
        avg_score = np.mean(scores)
        avg_max_tile = np.mean(max_tiles)
        win_rate = wins / episodes * 100
        
        print(f"\nTest Results ({episodes} episodes):")
        print(f"Average Score: {avg_score:.1f}")
        print(f"Average Max Tile: {avg_max_tile:.1f}")
        print(f"Win Rate: {win_rate:.1f}%")
        
        return scores, max_tiles, win_rate
    
    def plot_training_results(self, save_path="training_results.png"):
        """Plot training metrics."""
        plt.figure(figsize=(15, 5))
        
        # Plot scores
        plt.subplot(1, 3, 1)
        plt.plot(self.scores)
        plt.title('Game Score')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        
        # Plot max tiles
        plt.subplot(1, 3, 2)
        plt.plot(self.max_tiles)
        plt.title('Max Tile')
        plt.xlabel('Episode')
        plt.ylabel('Max Tile Value')
        
        # Plot win rate (moving average)
        plt.subplot(1, 3, 3)
        window_size = min(100, len(self.win_history))
        if window_size > 0:
            win_rate = [np.mean(self.win_history[max(0, i-window_size):i+1])*100 
                       for i in range(len(self.win_history))]
            plt.plot(win_rate)
            plt.title(f'Win Rate ({window_size}-ep moving avg)')
            plt.xlabel('Episode')
            plt.ylabel('Win Rate (%)')
            plt.ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
