from stable_baselines3.common.callbacks import BaseCallback
from tabulate import tabulate
from colorama import Fore, Style
import numpy as np
import subprocess
import os
import pandas as pd

# Function to display the progress of the D3QN RL algorithm
def show_rlib_d3qn_progress(iteration_index, result, show_each_agent=True):
    """
    Displays the progress of D3QN during training, showing rewards and losses per agent.

    Parameters:
    iteration_index (int): The current iteration number.
    result (dict): The results of the RL training, including rewards and learner stats.
    max_steps (int): The maximum number of steps for normalization.
    show_each_agent (bool): If True, displays rewards and losses for each agent. Defaults to True.

    Returns:
    tuple: Mean loss and mean reward across all agents.
    """
    # Print section header
    print(Fore.BLACK + "______________________________________" + Style.RESET_ALL)
    print(Fore.MAGENTA + f"ITERATION {iteration_index}" + Style.RESET_ALL)
    print(Fore.BLACK + "______________________________________" + Style.RESET_ALL)

    # Initialize cumulative values for loss and reward
    cum_loss = 0
    cum_reward = 0
    count = 0

    # Get policy rewards safely
    policy_rewards = result.get('env_runners', {}).get('policy_reward_mean', {})

    if not policy_rewards:
        print(Fore.RED + "TERMINATED EPISODE, NO REWARD" + Style.RESET_ALL)

    # Table headers for displaying agent stats
    headers = ["Agent", "Total Loss", "Policy Loss", "Value Function Loss", "Reward"]
    table_data = []

    # Loop through all agents and gather their stats
    for agent_id, stats in result.get('info', {}).get('learner', {}).items():
        learner_stats = stats.get('learner_stats', {})  # Prevent KeyError

        total_loss = learner_stats.get('total_loss', 0)
        policy_loss = learner_stats.get('policy_loss', 0)
        vf_loss = learner_stats.get('vf_loss', 0)

        # Get reward safely
        reward = policy_rewards.get(agent_id, 0)

        # Accumulate reward and loss for average calculation
        cum_reward += reward
        cum_loss += total_loss
        count += 1

        if show_each_agent:
            table_data.append([
                agent_id,
                f"{Fore.RED}{total_loss}{Style.RESET_ALL}",
                f"{Fore.YELLOW}{policy_loss}{Style.RESET_ALL}",
                f"{Fore.GREEN}{vf_loss}{Style.RESET_ALL}",
                f"{Fore.CYAN}{reward}{Style.RESET_ALL}"
            ])

    # Print the table if enabled
    if show_each_agent:
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Calculate and display the mean reward and mean loss across all agents
    mean_reward = (cum_reward / count) if count > 0 else cum_reward
    mean_loss = (cum_loss / count) if count > 0 else cum_loss

    print(Fore.MAGENTA + f"ITERATION {iteration_index} FINISHED, with Reward {mean_reward}, Loss {mean_loss}" + Style.RESET_ALL)
    print(Fore.BLACK + "______________________________________" + Style.RESET_ALL)

    return mean_loss, mean_reward  # Return average loss and reward



