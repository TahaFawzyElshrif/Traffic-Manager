# -*- coding: utf-8 -*-
import os
import sys
import platform
import warnings
import logging
from google.colab import drive
from numpy._core.defchararray import mod
import argparse


# Suppress warnings and TensorFlow logs
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger().setLevel(logging.ERROR)
#stderr_backup = sys.stderr
#sys.stderr = open(os.devnull, 'w') 

parser = argparse.ArgumentParser(description="Train and evaluate RL agent in SUMO")
parser.add_argument(
        "-p", "--project-path",
        type=str,
        required=True,
        help="Path to the project folder")

parser.add_argument(
        "--save-parameters",
        action="store_true", # bool
        help="Path to Save trained model parameters and normalization ")

parser.add_argument(
        "--optuna",
        action="store_true", # bool
        help="Is it normal experiment or hyperparameter experiment")

# Use Parser
parser_args = parser.parse_args()

path_project = parser_args.project_path
save_parameters = parser_args.save_parameters

enable_optuna = parser_args.optuna 

# ----------------------------
# Utility Functions
# ----------------------------
def is_colab():
    """Check if the code is running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

# ----------------------------
# Set Paths
# ----------------------------
if platform.system() == "Linux" and is_colab():
    path_main_folder = path_project + "/"

path_project_folder = path_main_folder + ""

yaml_file = path_project_folder + "config.yaml"
EXCEL_PATH = path_project_folder + ("OptunaHyper.xlsx" if enable_optuna else "sheet_full_environment_experiments.xlsx")
keys_file = path_project_folder + "keys.env"
path_info = path_project_folder + "info_road.csv"
log_file = "sumo_log.txt"

if not os.path.exists(path_project):
    print(f"File does not exist at: {path_project}")

# ----------------------------
# Import Packages
# ----------------------------
sys.path.append(path_project_folder)
from Callbacks import *
from models.d3qn import D3QNAgent
from data_parser import *
from Observations.sumo_obs import DefaultObservation
import SumoEnvSingleAgent
from Utils_reporting import *
from Utils_running_singleAgent import *
from rewards import *
from Connections import SumoConnection
from Connections.Connection import *

from dotenv import load_dotenv
import traci
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import NormalizeObservation, NormalizeReward
import optuna
import torch
import time
from colorama import Fore, Style
from stable_baselines3.common.evaluation import evaluate_policy
import yaml



# ----------------------------
# Load Parameters
# ----------------------------
reward_func = {
    'proposed_reward': reward_proposed,
    'literature_reward': reward_liter,
    'project_reward': reward_proj,
}

env_classes = {
    "HighGroupedSumoEnv": SumoEnvSingleAgent.HighGroupedSumoEnv,
    "GroupedSumoEnv": SumoEnvSingleAgent.GroupedSumoEnv,
    "SumoEnv": SumoEnvSingleAgent.SumoEnv,
}

with open(yaml_file, "r") as file:
    config = yaml.safe_load(file)

general_settings = config['general_settings']
experiment_settings_changable = config["experiment_settings"]['changable_settings']
experiment_settings_const = config["experiment_settings"]["const_settings"]
algorithm_settings = config["algorithms_settings"]

# General settings
is_gui = general_settings['is_gui']
see_progress_each = general_settings['see_progress_each']
enable_variation_action = general_settings["enable_variation_action"]
yellow_time = general_settings["yellow_time"]

# Constant experiment settings
max_steps = experiment_settings_const["max_steps"]
n_env = experiment_settings_const["n_env"]
durations = experiment_settings_const["durations"]
enable_gcd = experiment_settings_const["enable_gcd"]
n_episode_evaluation = experiment_settings_const["n_episode_evaluation"]
larger_evaluation = experiment_settings_const["larger_evaluation"]
larg_eval_enable = True

if enable_gcd:
    step_size, reduced_durations = gcd_and_reduced(durations)
else:
    step_size = 1
    reduced_durations = durations

# Load next row from Excel
next_sheet_row = ReadRow(EXCEL_PATH) if not enable_optuna else ReadRow(EXCEL_PATH,"Best Parameters")
next_sheet_row = clean_dict_values(next_sheet_row)

# Changable experiment settings
data_name = next_sheet_row['Area'] # the parameter only affect when not optimizing
n_epsiode = int(next_sheet_row['Episodes']) # the parameter only affect when not optimizing
ENV_NAME =next_sheet_row['Environment Type']
REWARD_TYPE = next_sheet_row['Reward']
seed = int(next_sheet_row['Seed']) # the parameter only affect when not optimizing
begin_time = larger_evaluation
n_step = int(next_sheet_row['Max Sumo Steps(s)'])
end_time = begin_time + n_step
algorithm = next_sheet_row['Algorithm'] 

precent_scale = .3 if enable_optuna else next_sheet_row['Traffic Scale'] 
sumo_traffic_scale = round(1+precent_scale,2) #int(10 * precent_scale)

EXPERIMENT_NAME = experiment_settings_changable["EXPERIMENT_NAME"]

# Data folder paths
if data_name == 'Mosheer':
    path_data_folder = path_project_folder + "AIST/data2_mosheerIsmail/"

        
elif data_name == "Bench(2waySingle)":
    path_data_folder = path_project_folder + "AIST/2waySingle/"
    begin_time = 0
    n_step = int(1e5 - begin_time) # 100000
    end_time = begin_time + n_step
    #larger_evaluation = 0 # no need for this in the benechmark
    larg_eval_enable = False
    n_episode_evaluation = 1
else:
    path_data_folder = path_project_folder + "AIST/data3_san_stefano/"

path_cfg = path_data_folder + "cfg.sumocfg"

MODEL_NAME = f"{data_name}_{REWARD_TYPE}_{algorithm}_{sumo_traffic_scale}_{seed}"
SAVE_PATH = path_project_folder+f"/OUTPUT/ENV_{MODEL_NAME}/" if save_parameters else ""
if save_parameters:
    os.makedirs(SAVE_PATH, exist_ok=True)


print(Fore.RED + f"CURRENT Experiment -- Data: {data_name} | Reward: {REWARD_TYPE} | "
                 f"Algorithm: {algorithm} | Episodes: {n_epsiode} | Begin {begin_time} , End: {end_time} ({n_step} seconds) | "
                 f"Traffic Scale: {sumo_traffic_scale} | Seed: {seed}" + Style.RESET_ALL)

# ----------------------------
# Initialize SUMO Connection
# ----------------------------
conn = SumoConnection.SumoConnection(path_cfg, step_size, log_file,begin_time=begin_time, end_time=end_time, seed=seed)
if  data_name == "Bench(Cologne)":
    # Restore it default
    conn.collision_mingap_factor = "-1"
    conn.weights_random_factor = "1"
    conn.time_to_teleport = 300
    conn.max_depart = -1

    yellow_time = 2

    reduced_durations = [5 ,22 ,35, 50]
def create_env(config_):
    """Create the simulation environment with given config."""
    env = env_classes[ENV_NAME](
        data_name=data_name,
        durations=reduced_durations,
        reward_fun=reward_func[REWARD_TYPE+"_reward"],
        step_size=step_size,
        obs_class=DefaultObservation,
        path_info=path_info,
        yellow_time=yellow_time,
        max_steps=max_steps,
        sumo_traffic_scale=sumo_traffic_scale,
        enable_variation_action=enable_variation_action,
        config=config_,
        seed=seed
    )
    env.data_path = path_data_folder
    env.see_progress_each = see_progress_each
    return env

set_global_conn(conn)
# ----------------------------
# Prepare Road Info (Optional, Only enable this code when adding new agent/area)
# ----------------------------
enable_editing = False
clear_file = False

if enable_editing:
    from data_parser import *
    from Utils_running_singleAgent import *

    path_info = path_project_folder + "info_road.csv"
    agent_ids = ["t"]
    agents_info = []

    for ag_i in agent_ids:
        lanes = traci.trafficlight.getControlledLanes(ag_i)
        direction_lanes = direction(lanes)
        out_lanes = [link[0][1] for link in traci.trafficlight.getControlledLinks(ag_i) if link]
        lanes_length = {lane: traci.lane.getLength(lane) for lane in lanes}
        edges = [traci.lane.getEdgeID(l) for l in lanes]

        agent_info = {
            'data': data_name,
            'agent_id': ag_i,
            'lanes': lanes,
            'out_lanes': out_lanes,
            'lanes_length': lanes_length,
            'edge': edges,
            'direction_lanes': direction_lanes
        }
        agents_info.append(agent_info)
        write_road_info(path_info, agents_info, clear_file=clear_file)
    print("Written " ,agents_info)
else:
    print("Skipped Writing Road Info")



# ----------------------------
# DQN Setup
# ----------------------------
dqn_settings = algorithm_settings['DQN']

exploration_initial_eps = dqn_settings["exploration_initial_eps"]
exploration_final_eps = dqn_settings["exploration_final_eps"]
exploration_fraction = dqn_settings["exploration_fraction"]
learning_rate = float(dqn_settings["learning_rate"])
gamma = dqn_settings["gamma"]
policy_kwargs = dict(net_arch=dqn_settings["policy_kwargs"]["net_arch"], activation_fn=torch.nn.ReLU)
batch_size = dqn_settings['batch_size']

# Reset connection and environment
conn.reset()
env = create_env({})



if algorithm == 'dqn':
    conn.reset()
    env = create_env({})
    if ("Bench" in  data_name):
        print(Fore.BLUE + "Begin Intialize DQN..." + Style.RESET_ALL)
        model = DQN(
            env=env,
            policy="MlpPolicy",
            learning_rate=0.001,
            learning_starts=0,
            train_freq=1,
            target_update_interval=500,
            exploration_initial_eps=0.05,
            exploration_final_eps=0.01,
            verbose=1,
        )
    else:
        print(Fore.BLUE + "Begin Intialize EPS - DQN..." + Style.RESET_ALL)

        model = EpsDQN(
            RMS_DQNPolicy,
            env,
            verbose=1,
            batch_size=batch_size,
            learning_rate=learning_rate,
            gamma=gamma,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            exploration_fraction=exploration_fraction,
            policy_kwargs=policy_kwargs,
            seed=seed
        )

    
    callback = Stable_RewardCallback(max_episodes=n_epsiode)
    time_before = time.time()
    model.learn(total_timesteps=1e11, callback=callback)
    time_after = time.time()
    rewards = callback.episode_rewards
    results_dict = env.last_run_dict
    env.close()

    print(Fore.BLUE + f"Begin Evaluating DQN On Same Time..."+ Style.RESET_ALL)
    evaluate_results= evaluate_policy(model, env, n_eval_episodes=n_episode_evaluation, return_episode_rewards=False)[0]
    #env.save(os.path.join(SAVE_PATH, "vecnormalize.pkl"))
    conn.reset()

else:
    print("Skipped DQN")

# ----------------------------
# PPO Setup and Training
# ----------------------------


if algorithm == 'PPO':
  if enable_optuna:
      print(Fore.BLUE + "Begin OPTUNA PPO..." + Style.RESET_ALL)
      import optuna

      n_tune_episode = 10
      n_trials = 20
      seeds = [0, 1, 2]  

     

      def objective(trial):
          learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3)
          gamma = trial.suggest_float('gamma', 0.9, 0.9999)
          gae_lambda = trial.suggest_float('gae_lambda', 0.8, 1.0)
          ent_coef = trial.suggest_float('ent_coef', 0.0, 0.1)
          clip_range = trial.suggest_float('clip_range', 0.1, 0.4)
          batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])
          net_arch = trial.suggest_categorical('net_arch', [32, 64, 128, 256, 512])

          policy_kwargs = dict(
              net_arch=[net_arch],
              activation_fn=torch.nn.ReLU
          )

          rewards_diffs = []

          for seed in seeds:
              env = create_env({})
              vec_env = DummyVecEnv([lambda: env])
              vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

              model = PPO(
                  "MlpPolicy", vec_env,
                  learning_rate=learning_rate,
                  gamma=gamma,
                  gae_lambda=gae_lambda,
                  ent_coef=ent_coef,
                  clip_range=clip_range,
                  batch_size=batch_size,
                  verbose=0,
                  policy_kwargs=policy_kwargs,
                  seed=seed
              )

              callback = Stable_RewardCallback(max_episodes=n_tune_episode)
              model.learn(total_timesteps=int(1e4), callback=callback)  # reduce steps for tuning

              rewards = callback.episode_rewards
              if len(rewards) >= 2:
                  rewards_diff = rewards[-1] - rewards[0]
                  rewards_diffs.append(rewards_diff)
                  print(Fore.BLUE + f"Seed {seed} epsiodes done in trial {trial.number+1}" + Style.RESET_ALL)


          avg_reward_diff = (sum(rewards_diffs) / len(rewards_diffs)) if (len(rewards_diffs)>0) else sum(rewards_diffs)

          print(Fore.GREEN + f"-------------Trial {trial.number+1} finished, avg reward delta = {avg_reward_diff:.2f}-----------" + Style.RESET_ALL)
          return avg_reward_diff

      # Create Optuna study
      study = optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=42))
      study.optimize(objective, n_trials=n_trials)

      print("Best Hyperparameters:", study.best_params)

      WriteRow({
          "Best Parameters":str(study.best_params), 
          "Best Value":str(study.best_value)
        },EXCEL_PATH,"Best Parameters")

      print(Fore.CYAN + f"Written completed. " + Style.RESET_ALL)
      env.close()
  
  else:
      print("Skipped Optimizing PPO")

      print(Fore.BLUE + "Initializing PPO..." + Style.RESET_ALL)
      env = create_env({})
      vec_env = DummyVecEnv([lambda: env])
      vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

      ppo_settings = algorithm_settings['PPO'][f"{data_name}_{ENV_NAME}_{REWARD_TYPE}_reward"]
      policy_kwargs = {"net_arch": [ppo_settings["net_arch"]], "activation_fn": torch.nn.ReLU}

      model = PPO(
          "MlpPolicy", vec_env,
          learning_rate=ppo_settings['learning_rate'],
          gamma=ppo_settings['gamma'],
          gae_lambda=ppo_settings['gae_lambda'],
          ent_coef=ppo_settings['ent_coef'],
          clip_range=ppo_settings['clip_range'],
          batch_size=ppo_settings['batch_size'],
          verbose=0,
          policy_kwargs=policy_kwargs,
          seed=seed
      )
      callback = Stable_RewardCallback(max_episodes=n_epsiode)

      print(Fore.BLUE + "Begin Training PPO..." + Style.RESET_ALL)
      time_before = time.time()
      model.learn(total_timesteps=1e9, callback=callback)
      time_after = time.time()
      rewards = callback.episode_rewards
      results_dict = env.last_run_dict

      print(Fore.BLUE + f"Begin Evaluating PPO On Same Time..."+ Style.RESET_ALL)
      evaluate_results= evaluate_policy(model, env, n_eval_episodes=n_episode_evaluation, return_episode_rewards=False)[0]
      vec_env.save(os.path.join(SAVE_PATH, "vecnormalize.pkl"))
      conn.reset()
      
      if larg_eval_enable:
          print(Fore.BLUE + f"Begin Evaluating PPO For Larger Time {larger_evaluation/60} M..."+ Style.RESET_ALL)
          
          conn.end_time =  conn.begin_time + larger_evaluation

          eval_env = create_env({})
          vec_eval_env = DummyVecEnv([lambda: eval_env])
          vec_eval_env = VecNormalize(vec_eval_env, norm_obs=True, norm_reward=False)
          vec_eval_env = VecNormalize.load(os.path.join(SAVE_PATH, "vecnormalize.pkl"), vec_eval_env)

          obs = vec_eval_env.reset()
          done = False
          c_reward = 0

          while not done:
              action, _ = model.predict(obs, deterministic=True)
              obs, reward, done, _ = vec_eval_env.step(action)
              c_reward += reward

          eval_last_run_dict = eval_env.last_run_dict
          print(f"Eval METRICES {eval_last_run_dict}")
          vec_eval_env.close()

      if save_parameters:
            model.save(os.path.join(SAVE_PATH, "ppo_model"))
            print(f"MODEL IS SAVED AT {os.path.join(SAVE_PATH, "ppo_model")}")

      else:
            os.remove(os.path.join(SAVE_PATH, "vecnormalize.pkl"))
            print(f"REMOVED TEMP FILE  {os.path.join(SAVE_PATH, "vecnormalize.pkl")}")

# ----------------------------
# D3QN Setup and Training
# ----------------------------
if algorithm == "D3QN":
    if enable_optuna:
        print(Fore.BLUE + f"Begin OPTUNA D3QN..."+ Style.RESET_ALL)


       
        n_tune_epsiode = 10
        n_trials = 20

        def objective(trial):
            # Sample hyperparameters
            learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3)
            gamma = trial.suggest_float('gamma', 0.9, 0.9999)
            tau = trial.suggest_float('tau', 0.8, 1.0)
            l2_reg = trial.suggest_float('l2_reg', 0.001, 0.01)
            epsilon_decay = trial.suggest_float('epsilon_decay', 0.0001, 0.4)
            batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])

            reward_diffs = []
            seeds=[0,1,2]

            for seed in seeds:
                env = create_env({})
                env = NormalizeObservation(env, epsilon=1e-8)

                state_size = env.observation_space.shape
                num_actions = env.action_space.n

                conn.reset()
                env.reset()
                conn.seed = seed
                env.env.seed = seed
                agent = D3QNAgent(
                    env=env,
                    state_size=state_size,
                    num_actions=num_actions,
                    memory_size=100000,
                    batch_size=batch_size,
                    gamma=gamma,
                    epsilon_start=1.0,
                    epsilon_min=0.01,
                    epsilon_decay=epsilon_decay,
                    learning_rate=learning_rate,
                    tau=tau,
                    update_freq=4,
                    l2_reg=l2_reg,
                    random_state=seed
                )


                training_results = agent.train(
                    num_episodes=n_tune_epsiode,
                    max_steps_per_episode=200,
                    num_points_for_average=100,
                    log_interval=10
                )

                rewards_diff = training_results['rewards'][-1] - training_results['rewards'][0]
                reward_diffs.append(rewards_diff)
                print(Fore.BLUE + f"Seed {seed} epsiodes done in trial {trial.number+1}" + Style.RESET_ALL)

            avg_reward_diff = np.mean(reward_diffs)
            print(Fore.GREEN + f"Trial {trial.number+1} Finished, avg_derivative: {avg_reward_diff:.2f}" + Style.RESET_ALL)
            return avg_reward_diff


        # Create an Optuna study
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        # Print best hyperparameters
        print("Best Hyperparameters:", study.best_params)
        
        WriteRow({
          "Best Parameters":str(study.best_params), 
          "Best Value":str(study.best_value)
        },EXCEL_PATH,"Best Parameters")

        print(Fore.CYAN + f"Written completed. " + Style.RESET_ALL)
        env.close()
    else:
        print("Skipped Optuna")

 
        env = create_env({})
        env = NormalizeObservation(env, epsilon=1e-8)
        state_size = env.observation_space.shape
        num_actions = env.action_space.n

        d3qn_settings = algorithm_settings['D3QN'][f"{data_name}_{ENV_NAME}_{REWARD_TYPE}_reward"]
        agent = D3QNAgent(
            env=env,
            state_size=state_size,
            num_actions=num_actions,
            memory_size=100000,
            batch_size=d3qn_settings['batch_size'],
            gamma=d3qn_settings['gamma'],
            epsilon_start=1.0,
            epsilon_min=0.01,
            epsilon_decay=d3qn_settings['epsilon_decay'],
            learning_rate=d3qn_settings['learning_rate'],
            tau=d3qn_settings['tau'],
            update_freq=4,
            l2_reg=d3qn_settings['l2_reg'],
            random_state=seed
        )

        print(Fore.BLUE + "Begin Training D3QN..." + Style.RESET_ALL)
        time_before = time.time()
        training_results = agent.train(
            num_episodes=n_epsiode,
            max_steps_per_episode=max_steps,
            num_points_for_average=100,
            log_interval=1
        )
        time_after = time.time()
        rewards = training_results['rewards']
        results_dict = env.env.last_run_dict

        # Evaluation for same time
        
        print(Fore.BLUE + f"Begin Evaluating D3QN On Same Time..."+ Style.RESET_ALL)
        evaluate_results = agent.evaluate(num_episodes=n_episode_evaluation)

        # Evaluation for larger time       
        conn.reset()

        if larg_eval_enable:
            print(Fore.BLUE + f"Begin Evaluating D3QN For Larger Time {larger_evaluation/60} M..."+ Style.RESET_ALL)
            conn.end_time  = conn.begin_time + larger_evaluation

            eval_env = create_env({})
            eval_env = NormalizeObservation(eval_env, epsilon=1e-8)    
            
            obs,_ = eval_env.reset()
            done = False
            c_reward = 0
            while (not done) :
                action = agent.get_action(obs) 
                obs, reward, done,_, info = eval_env.step(action) # Can be negative as normalized
                c_reward += reward
                if done:
                  break

            conn.close()
            eval_env.close()
            eval_last_run_dict = eval_env.env.last_run_dict
            print("Eval test ", eval_last_run_dict)
        if save_parameters:
            agent.save_model(os.path.join(SAVE_PATH, "d3qn_model.keras"))
            print(f"MODEL IS SAVED AT {os.path.join(SAVE_PATH, "d3qn_model")}")

        

# ----------------------------
# Evaluation & Saving Results
# ----------------------------
if enable_optuna:
    pass
else:
    time_diff = time_after - time_before
    last_cumulative_reward = round(rewards[-1], 3)
    derivative = rewards[-1] - rewards[0]

    if save_parameters:
        import matplotlib.pyplot as plt     
        plt.plot(rewards)
        plt.savefig(SAVE_PATH+f"reward_during_training.png")  


    
    print(Fore.GREEN + f"Training time: {round(time_diff,3)} sec ({round(time_diff/60,3)} min)" + Style.RESET_ALL)
    print(Fore.MAGENTA + f"Cumulative Reward of last episode: {last_cumulative_reward} | Reward: {REWARD_TYPE}" + Style.RESET_ALL)

    for key, value in results_dict.items():
        print(Fore.CYAN + f"{key}: {round(value,3)}" + Style.RESET_ALL)
    
    WriteRow({
        "Reward of Last Episode": last_cumulative_reward,
        "Derivative of Reward": derivative,
        "Waiting Time (s)": results_dict['waiting_time'],
        "Speed (m/s)": results_dict['speed'],
        "Depart Delay (s)": results_dict['depart_delay'],
        "Time Loss (s)": results_dict['time_loss'],
        "Waiting Car": results_dict['waiting_vehicles'],

        "Average Reward for Evaluated Episodes With Same Time":evaluate_results,
        "Waiting Car on Final Test":eval_last_run_dict['waiting_vehicles'] if  (larg_eval_enable) else 0,
        "Time Loss (s) on Final Test":eval_last_run_dict['time_loss'] if  (larg_eval_enable) else 0,
        "Depart Delay (s) on Final Test": eval_last_run_dict['depart_delay'] if  (larg_eval_enable) else 0,
        "Speed (m/s) on Final Test":eval_last_run_dict['speed'] if  (larg_eval_enable) else 0,
        "Waiting Time (s) on Final Test":eval_last_run_dict['waiting_time'] if  (larg_eval_enable) else 0,
        "Reward on Final Test": c_reward if  (larg_eval_enable) else 0,


        "Total Time Of Training (M)": round(time_diff/60, 3),
        "Device": "colab"
    }, EXCEL_PATH)
    
    print(Fore.CYAN + f"Written completed. Finished {count_full_rows(EXCEL_PATH)} Experiments" + Style.RESET_ALL)

    conn.close()
    env.close()
    print(Fore.CYAN + "---------------------------------------" + Style.RESET_ALL)
