import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
from collections import deque
import os
import matplotlib.pyplot as plt
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures

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
        self.learning_rate = .0005
        self.gamma = 0.999  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9999
        
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
        self.reward_history = []
        self.max_tiles = []
        self.win_history = []
        self.epsilon_history = []
        
        # Create plots directory
        os.makedirs("../training_plots", exist_ok=True)
    
    # @timer
    def _build_model(self):
        """Create a CNN model for Q-value approximation."""
        inputs = layers.Input(shape=self.state_shape)
        
        # Reshape to add channel dimension for CNN
        x = layers.Reshape((*self.state_shape, 1))(inputs)
        
        # CNN layers with batch normalization
        x = layers.Conv2D(64, (2, 2), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (2, 2), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (2, 2), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Flatten and dense layers with batch normalization
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
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
    
    @tf.function
    def _train_on_batch(self, states, targets):
        with tf.GradientTape() as tape:
            predictions = self.model(states, training=True)
            mse = tf.keras.losses.MeanSquaredError()
            loss = mse(targets, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([exp[0] for exp in minibatch])
        actions = np.array([exp[1] for exp in minibatch])
        rewards = np.array([exp[2] for exp in minibatch])
        next_states = np.array([exp[3] for exp in minibatch])
        dones = np.array([exp[4] for exp in minibatch])
        
        current_q_values = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)

        targets = current_q_values.copy()
        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        self._train_on_batch(states, targets)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
    def load(self, name):
        """Load model weights."""
        self.model.load_weights(name)
        self.update_target_model()
    
    def save(self, name):
        """Save model weights."""
        self.model.save_weights(name)
    
    def generate_training_plots(self, episode, plot_every, save_dir="../training_plots"):
        """Generate plots during training to track progress."""
        # Only generate plots at specified intervals or at the end
        if episode % plot_every != 0 and episode != 0:
            return
            
        # Don't generate plots if we don't have any data yet
        if not self.scores:
            print("Not enough data to generate plots yet. Skipping plot generation.")
            return

        # Create figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Create episode index for x-axis
        episodes_range = list(range(len(self.scores)))
        
        # Plot scores
        axs[0, 0].plot(episodes_range, self.scores)
        axs[0, 0].set_title('Game Score')
        axs[0, 0].set_xlabel('Episode')
        axs[0, 0].set_ylabel('Score')
        
        # Add a 100-episode moving average to the score plot if there are enough episodes
        if len(self.scores) >= 100:
            moving_avg = np.convolve(self.scores, np.ones(100)/100, mode='valid')
            axs[0, 0].plot(range(99, len(self.scores)), moving_avg, 'r', label='100-ep Moving Avg')
            axs[0, 0].legend()
        
        # Plot max tiles
        axs[0, 1].plot(episodes_range, self.max_tiles)
        axs[0, 1].set_title('Max Tile')
        axs[0, 1].set_xlabel('Episode')
        axs[0, 1].set_ylabel('Max Tile Value')
        
        # Plot win rate (moving average)
        window_size = min(100, len(self.win_history))
        if window_size > 0:
            win_rate = [np.mean(self.win_history[max(0, i-window_size):i+1])*100 
                       for i in range(len(self.win_history))]
            axs[1, 0].plot(episodes_range, win_rate)
            axs[1, 0].set_title(f'Win Rate ({window_size}-ep moving avg)')
            axs[1, 0].set_xlabel('Episode')
            axs[1, 0].set_ylabel('Win Rate (%)')
            axs[1, 0].set_ylim(0, 100)
        
        # Plot epsilon decay
        axs[1, 1].plot(episodes_range, self.epsilon_history)
        axs[1, 1].set_title('Exploration Rate (Epsilon)')
        axs[1, 1].set_xlabel('Episode')
        axs[1, 1].set_ylabel('Epsilon')
        axs[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        filename = os.path.join(save_dir, f"training_progress_ep{episode}.png")
        plt.savefig(filename)
        plt.close(fig)
        
        print(f"Training progress plot saved to {filename}")
    
    def train_headless(self, env, episodes=10000, save_every=500, print_every=10, 
                      plot_every=100, checkpoint_dir="../checkpoints"):
        """
        Train the agent on a headless environment for fast training.
        
        Parameters:
        - env: The game environment (must implement reset(), step(), get_valid_moves())
        - episodes: Number of episodes to train for
        - save_every: Save the model every X episodes
        - print_every: Print progress every X episodes
        - plot_every: Generate and save plots every X episodes
        - checkpoint_dir: Directory to save checkpoints
        
        Returns:
        - Training statistics (scores, max_tiles, win_history)
        """
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Ensure plot_every is not too frequent (minimum 10 episodes)
        plot_every = max(10, plot_every)
        
        start_time = time.time()
        # print the hyperparameters
        print(f"Training hyperparameters: \n"
                f"  - Episodes: {episodes}\n"
                f"  - Save every: {save_every}\n"
                f"  - Print every: {print_every}\n"
                f"  - Plot every: {plot_every}\n"
                f"  - Checkpoint directory: {checkpoint_dir}\n"
                f"  - Epsilon decay: {self.epsilon_decay}\n"
                f"  - Epsilon min: {self.epsilon_min}\n"
                f"  - Learning rate: {self.learning_rate}\n"
                f"  - Batch size: {self.batch_size}\n"
                f"  - Memory size: {len(self.memory)}\n"
                f"  - Gamma: {self.gamma}\n"
                f"  - Target model update frequency: 500 steps\n"
        )
        
        for episode in range(episodes):
            # Reset the environment
            state = env.reset()
            total_reward = 0
            done = False
            
            # Track episode statistics
            steps = 0
            max_tile = 0
            
            while not done:
                # Get valid moves
                valid_moves = env.get_valid_moves()
                
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
                if steps % 4 == 0:  # Learn every 4 steps
                    self.replay()
                
                # Update target model periodically
                if steps % 500 == 0:
                    self.update_target_model()
                    
                steps += 1
            
            # Record episode results
            self.train_episodes += 1
            score = info.get('score', 0)
            

            # print(score)
            self.scores.append(score)
            self.max_tiles.append(max_tile)
            self.win_history.append(1 if info['won'] else 0)
            self.epsilon_history.append(self.epsilon)
            self.reward_history.append(total_reward)
            
            # Update best score
            if info['score'] > self.best_score:
                self.best_score = info['score']
                self.save(os.path.join(checkpoint_dir, "best_model.weights.h5"))

            # update best reward
            if total_reward > max(self.reward_history, default=0):
                self.save(os.path.join(checkpoint_dir, "best_reward_model.weights.h5"))
            
            # Print progress
            if episode % print_every == 0 or episode == episodes - 1:
                avg_score = np.mean(self.scores[-print_every:]) if self.scores else 0
                avg_max_tile = np.mean(self.max_tiles[-print_every:]) if self.max_tiles else 0
                win_rate = np.mean(self.win_history[-print_every:]) * 100 if self.win_history else 0
                elapsed = time.time() - start_time
                avg_reward = np.mean(self.reward_history[-print_every:]) if self.reward_history else 0  
                print(f"Episode: {episode}/{episodes} | "
                      f"Score: {info['score']} | "
                      f"Reward: {total_reward} | "
                      f"Max Tile: {max_tile} | "
                      f"Steps: {steps} | "
                      f"Avg Score: {avg_score:.1f} | "
                      f"Avg Reward: {avg_reward:.1f} | "
                      f"Win Rate: {win_rate:.1f}% | "
                      f"Epsilon: {self.epsilon:.4f} | "
                      f"Elapsed: {elapsed:.1f}s")
            
            # Generate training plots - but only after we have collected some data
            if episode >= 10:  # Wait for at least 10 episodes before first plot
                self.generate_training_plots(episode, plot_every)
            
            # Save model periodically
            if episode > 0 and episode % save_every == 0:
                self.save(os.path.join(checkpoint_dir, f"model_ep{episode}.weights.h5"))
        
        # Final save
        self.save(os.path.join(checkpoint_dir, "final_model.weights.h5"))
        
        # Final plot
        self.generate_training_plots(episodes, 1)
        
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
        """Plot training metrics and save to file."""
        plt.figure(figsize=(15, 10))
        
        # Create episode index for x-axis
        episodes_range = list(range(len(self.scores)))
        
        # Plot scores
        plt.subplot(2, 2, 1)
        plt.plot(episodes_range, self.scores)
        plt.title('Game Score')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        
        # Add moving average if we have enough data
        if len(self.scores) >= 100:
            moving_avg = np.convolve(self.scores, np.ones(100)/100, mode='valid')
            plt.plot(range(99, len(self.scores)), moving_avg, 'r', label='100-ep Moving Avg')
            plt.legend()
        
        # Plot max tiles
        plt.subplot(2, 2, 2)
        plt.plot(episodes_range, self.max_tiles)
        plt.title('Max Tile')
        plt.xlabel('Episode')
        plt.ylabel('Max Tile Value')
        
        # Plot win rate (moving average)
        plt.subplot(2, 2, 3)
        window_size = min(100, len(self.win_history))
        if window_size > 0:
            win_rate = [np.mean(self.win_history[max(0, i-window_size):i+1])*100 
                       for i in range(len(self.win_history))]
            plt.plot(episodes_range, win_rate)
            plt.title(f'Win Rate ({window_size}-ep moving avg)')
            plt.xlabel('Episode')
            plt.ylabel('Win Rate (%)')
            plt.ylim(0, 100)
        
        # Plot epsilon decay
        plt.subplot(2, 2, 4)
        plt.plot(episodes_range, self.epsilon_history)
        plt.title('Exploration Rate (Epsilon)')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
