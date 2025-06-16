from colorama import Fore, Style
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.torch_layers import create_mlp
from torch.optim import RMSprop
import torch.nn as nn
import gym
from stable_baselines3 import DQN
from stable_baselines3.dqn.dqn import DQN as SB3_DQN
import math
from stable_baselines3.common.callbacks import BaseCallback

# Callback class to track rewards during RL training
class Stable_RewardCallback(BaseCallback):
    def __init__(self, max_episodes):
        """
        Initializes the callback for tracking rewards and losses during training.

        Parameters:
        max_episodes (int): The maximum number of episodes to track.
        """
        super().__init__()
        self.max_episodes = max_episodes
        self.episode_rewards = []  
        self.episode_reward = 0 # Cumulative reward for each episode
        self.episode_count = 0


    def _on_step(self) -> bool:
        """
        Tracks the reward and loss for each step during training. When an episode ends,
        it stores the cumulative reward and loss, and prints the results.

        Returns:
        bool: True to continue training, False to stop after max episodes.
        """
        self.episode_reward += self.locals["rewards"][0]
        
        if self.locals["dones"][0]:
            # Store the cumulative reward and loss for the episode
            self.episode_rewards.append(self.episode_reward)

            # Print the result for the current episode
            print(Fore.MAGENTA + f"ITERATION {self.episode_count} FINISHED, with Reward {self.episode_reward}" + Style.RESET_ALL)

            # Reset reward and loss for the next episode
            self.episode_loss = 0
            self.episode_reward = 0

            # Increment episode count
            self.episode_count += 1

            # Stop training if the maximum number of episodes is reached
            if self.episode_count >= self.max_episodes:
                return False  # Stops training

        return True

# Custom Policy with RMSprop
class RMS_DQNPolicy(DQNPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        # Build the Q-network using the parent method
        super()._build(lr_schedule)

        # Replace the optimizer with RMSprop
        self.optimizer = RMSprop(self.parameters(), lr=lr_schedule(1))

class EpsDQN(DQN): #DQN
    def __init__(self, *args, min_epsilon=0.01, max_epsilon=1.0, decay_rate=0.001, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.decay_rate = decay_rate
        self.custom_step = 0  # separate counter to track steps

    def _update_learning_rate(self, optimizer):
        # override default schedule if needed
        pass

    def train(self, gradient_steps: int, batch_size: int = 64):
        # update epsilon manually here
        self.custom_step += 1
        self.exploration_rate = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * math.exp(-self.decay_rate * self.custom_step)

        # call the original train
        return super().train(gradient_steps, batch_size)