import numpy as np
import tensorflow as tf
import random
import time
from collections import deque
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, RepeatVector, Reshape, Add, Subtract
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE
from dataclasses import dataclass
from colorama import Fore, Style

@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool

class D3QNAgent:
    """
    Dueling Double Deep Q-Network (D3QN) Agent for Gymnasium environments
    """
    def __init__(
        self,
        env,
        state_size,
        num_actions,
        memory_size=100000,
        batch_size=64,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        learning_rate=0.001,
        tau=0.001,
        update_freq=4,
        hidden_layers=[64, 64],
        l2_reg=0.001,
        random_state=None
    ):
        """
        Initialize the D3QN agent
        
        Args:
            env: Gymnasium environment
            state_size: Shape of state observations
            num_actions: Number of possible actions
            memory_size: Max size of replay buffer
            batch_size: Batch size for training
            gamma: Discount factor
            epsilon_start: Starting value for epsilon
            epsilon_min: Minimum value for epsilon
            epsilon_decay: Decay rate for epsilon
            learning_rate: Learning rate for optimizer
            tau: Rate of target network update
            update_freq: How often to update the network
            hidden_layers: List of hidden layer sizes
            l2_reg: L2 regularization factor
            random_state: Random seed for reproducibility
        """
        # Debug info
        print(f"Environment observation space: {env.observation_space}")
        print(f"Environment action space: {env.action_space}")
        print(f"State size provided: {state_size}")
        
        # Environment parameters
        self.env = env
        self.state_size = state_size  
        self.num_actions = num_actions
        
        # Learning parameters
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.tau = tau
        self.update_freq = update_freq
        self.hidden_layers = hidden_layers
        self.l2_reg = l2_reg
        
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
            tf.random.set_seed(random_state)
            random.seed(random_state)
        
        # Initialize memory buffer
        self.memory_buffer = deque(maxlen=memory_size)
        
        # Build neural networks
        self.q_network = self.build_dueling_model()
        self.target_q_network = self.build_dueling_model()
        
        # Compile the model
        self.optimizer = Adam(learning_rate=learning_rate)
        
        # Initialize target network weights
        self.target_q_network.set_weights(self.q_network.get_weights())
        
        # Training metrics
        self.total_point_history = []
        self.loss_history = []
        self.step_count = 0
        
    def build_dueling_model(self):
        """
        Build a dueling network architecture for D3QN
        """
        # Handle different input shapes
        if isinstance(self.state_size, int):
            input_shape = (self.state_size,)
        else:
            input_shape = self.state_size
            
        state_input = Input(shape=input_shape)
        x = state_input
        
        # Hidden layers with L2 regularization
        for units in self.hidden_layers:
            x = Dense(
                units=units, 
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)
            )(x)
        
        # Value stream
        value_stream = Dense(
            units=self.hidden_layers[-1]//2, 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)
        )(x)
        value = Dense(1)(value_stream)
        
        # Advantage stream
        advantage_stream = Dense(
            units=self.hidden_layers[-1]//2, 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)
        )(x)
        advantage = Dense(self.num_actions)(advantage_stream)
        
        # Combine value and advantage streams using built-in layers
        # First, broadcast value to same shape as advantage
        value_broadcast = RepeatVector(int(self.num_actions))(value)

        value_broadcast = Reshape((self.num_actions,))(value_broadcast)
        
        # Calculate advantage mean and center it
        advantage_mean = Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True),output_shape=(1,))(advantage)
        advantage_centered = Subtract()([advantage, advantage_mean])
        
        # Add value and centered advantage
        q_values = Add()([value_broadcast, advantage_centered])
        
        return Model(inputs=state_input, outputs=q_values)

    def get_action(self, state):
        """
        Select an action using epsilon-greedy policy
        """
        # Ensure proper shape for the network
        if isinstance(state, np.ndarray) and state.ndim < 2:
            state = np.expand_dims(state, axis=0)
            
        if random.random() > self.epsilon:
            q_values = self.q_network(state)
            return np.argmax(q_values.numpy()[0])
        else:
            return random.randint(0, self.num_actions - 1)
    
    def store_experience(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer
        """
        self.memory_buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample_experiences(self):
        """
        Sample a batch of experiences from replay buffer
        """
        experiences = random.sample(self.memory_buffer, k=self.batch_size)
        
        states = tf.convert_to_tensor(
            np.array([e.state for e in experiences]), dtype=tf.float32
        )
        actions = tf.convert_to_tensor(
            np.array([e.action for e in experiences]), dtype=tf.int32
        )
        rewards = tf.convert_to_tensor(
            np.array([e.reward for e in experiences]), dtype=tf.float32
        )
        next_states = tf.convert_to_tensor(
            np.array([e.next_state for e in experiences]), dtype=tf.float32
        )
        done_vals = tf.convert_to_tensor(
            np.array([e.done for e in experiences]).astype(np.uint8),
            dtype=tf.float32,
        )
        
        return (states, actions, rewards, next_states, done_vals)
    
    def compute_loss(self, experiences):
        """
        Compute the D3QN loss
        """
        states, actions, rewards, next_states, done_vals = experiences
        states = tf.squeeze(states, axis=1)
        next_states = tf.squeeze(next_states, axis=1)
        
        # Double Q-learning part
        # Use online network to select actions
        q_values_next = self.q_network(next_states)
        best_actions = tf.argmax(q_values_next, axis=1)
        
        # Use target network to evaluate actions
        next_q_values = self.target_q_network(next_states)
        next_q_values_for_best_actions = tf.gather_nd(
            next_q_values, 
            tf.stack([tf.range(next_q_values.shape[0]), tf.cast(best_actions, tf.int32)], axis=1)
        )
        
        # Compute targets
        y_targets = rewards + (self.gamma * next_q_values_for_best_actions * (1 - done_vals))
        
        # Get Q-values for actions taken
        q_values = self.q_network(states)
        q_values_for_actions = tf.gather_nd(
            q_values, 
            tf.stack([tf.range(q_values.shape[0]), actions], axis=1)
        )
        
        # Compute loss
        loss = MSE(q_values_for_actions, y_targets)
        return loss
    
    def update_networks(self):
        """
        Update networks if enough experiences are available
        """
        if len(self.memory_buffer) < self.batch_size:
            return 0
        
        # Sample a batch of experiences
        experiences = self.sample_experiences()
        
        # Compute loss and apply gradients
        with tf.GradientTape() as tape:
            loss = self.compute_loss(experiences)
        
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        
        # Soft update target network
        for target_weights, q_net_weights in zip(self.target_q_network.weights, self.q_network.weights):
            target_weights.assign(self.tau * q_net_weights + (1.0 - self.tau) * target_weights)
        
        return loss.numpy()
    
    def decay_epsilon(self):
        """
        Decay epsilon according to schedule
        """
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)
    
    def train(self, num_episodes, max_steps_per_episode=200, num_points_for_average=100, log_interval=10):
        """
        Train the agent for a specific number of episodes
        
        Args:
            num_episodes: Number of episodes to train
            max_steps_per_episode: Maximum steps per episode
            num_points_for_average: Number of points to use for averaging
            log_interval: How often to log training information
        """
        start_time = time.time()
        
        for episode in range(num_episodes):
            # Reset environment (Gymnasium style)
            state, _ = self.env.reset()  # Updated for Gymnasium which returns (obs, info)
            
            # Ensure state has correct shape
            if isinstance(state, np.ndarray) and state.ndim < 2:
                state = np.expand_dims(state, axis=0)
                
            total_reward = 0
            episode_loss = []
            
            for step in range(max_steps_per_episode):
                # Select action
                action = self.get_action(state)
                
                # Take action (Gymnasium style)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated  # Combine terminal conditions
                
                # Ensure next_state has correct shape
                if isinstance(next_state, np.ndarray) and next_state.ndim < 2:
                    next_state = np.expand_dims(next_state, axis=0)
                
                # Store experience
                self.store_experience(state, action, reward, next_state, done)
                
                # Update networks periodically
                self.step_count += 1
                if self.step_count % self.update_freq == 0:
                    loss = self.update_networks()
                    episode_loss.append(loss)
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            # Store total reward for this episode
            self.total_point_history.append(total_reward)
            
            # Store average loss for this episode if available
            if episode_loss:
                self.loss_history.append(np.mean(episode_loss))
            
            # Decay epsilon
            self.decay_epsilon()
            
            # Log progress
            if (episode + 1) % log_interval == 0:
                avg_reward = np.mean(self.total_point_history[-num_points_for_average:])
                avg_loss = np.mean(self.loss_history[-num_points_for_average:]) if self.loss_history else 0
                
                
                print(Fore.CYAN +  f"\nEpisode {episode + 1}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Epsilon: {self.epsilon:.4f} | "
                      f"Avg Loss: {avg_loss:.6f}"+ Style.RESET_ALL)

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        return {
            'rewards': self.total_point_history,
            'losses': self.loss_history,
            'final_avg_reward': np.mean(self.total_point_history[-num_points_for_average:])
        }
    
    def save_model(self, filepath):
        """
        Save the trained model
        """
        self.q_network.save(filepath)
        
    def load_model(self, filepath):
        """
        Load a trained model
        """
        self.q_network = tf.keras.models.load_model(filepath)
        self.target_q_network.set_weights(self.q_network.get_weights())
    
    def evaluate(self, num_episodes=10, render=False):
        """
        Evaluate the agent without exploration
        """
        total_rewards = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()  # Updated for Gymnasium
            
            # Ensure state has correct shape
            if isinstance(state, np.ndarray) and state.ndim < 2:
                state = np.expand_dims(state, axis=0)
                
            episode_reward = 0
            done = False
            
            while not done:
                if render:
                    self.env.render()
                
                # Always choose best action (no exploration)
                q_values = self.q_network(state)
                action = np.argmax(q_values.numpy()[0])
                
                # Updated for Gymnasium
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                done = terminated or truncated
                
                # Ensure next_state has correct shape
                if isinstance(next_state, np.ndarray) and next_state.ndim < 2:
                    next_state = np.expand_dims(next_state, axis=0)
                    
                episode_reward += reward
                state = next_state
            
            total_rewards.append(episode_reward)
            
        avg_reward = np.mean(total_rewards)
        print(f"Evaluation over {num_episodes} episodes: Average Reward = {avg_reward:.2f}")
        
        return avg_reward