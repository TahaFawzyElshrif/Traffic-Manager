# -*- coding: utf-8 -*-
from google.colab import drive
import os
import platform
import sys
import warnings
import logging

# Clean unnecessary prints.
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  
sys.stderr = open(os.devnull, 'w')
logging.getLogger().setLevel(logging.ERROR)  

path_project = sys.argv[1]

"""### Set Main Pathes"""

def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False



if platform.system() == "Linux":
    if is_colab():
      path_main_folder = path_project + "/"



path_project_folder = path_main_folder + ""

yaml_file = path_project_folder + "config.yaml"
EXCEL_PATH = path_main_folder +"sheet_full_environment_experiments.xlsx"
keys_file = path_project_folder + "keys.env"
path_info = path_main_folder + "info_road.csv"
log_file = "sumo_log.txt"



if os.path.exists(path_project):
  pass
  #print(f"File exists at: {path_data_folder}")
else:
  print(f"File does not exist at: {path_project}")



"""### Import Packages"""

sys.path.append(path_project_folder)
from Connections import SumoConnection
from Connections.Connection import *
from dotenv import load_dotenv
import traci
import gymnasium as gym
from numpy import inf
import numpy as np
import SumoEnvSingleAgent
from Utils_reporting import *
from Utils_running_singleAgent import *
from rewards import *
from stable_baselines3 import PPO ,DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch
import time
from Callbacks import *
from models.d3qn import D3QNAgent
import yaml
from data_parser import *
from Observations.sumo_obs import DefaultObservation
from stable_baselines3.common.evaluation import evaluate_policy



"""### Load Parameters"""

reward_func = {
'proposed_reward':reward_proposed,
'literature':reward_liter,
'project_reward':reward_proj,
}
env_classes = {
    "HighGroupedSumoEnv": SumoEnvSingleAgent.HighGroupedSumoEnv,
    "GroupedSumoEnv": SumoEnvSingleAgent.GroupedSumoEnv ,
    "SumoEnv": SumoEnvSingleAgent.SumoEnv,
}


# Load YAML file
# if something modified ,just rerun this cell
with open(yaml_file, "r") as file:
    config = yaml.safe_load(file)

general_settings = config['general_settings']
experiment_settings_changable = config["experiment_settings"]['changable_settings']
experiment_settings_const = config["experiment_settings"]["const_settings"]
algorithm_settings=config["algorithms_settings"]

is_gui = general_settings['is_gui']
see_progress_each = general_settings['see_progress_each']
enable_variation_action = general_settings["enable_variation_action"]
yellow_time = general_settings["yellow_time"]

# Access specific parameters (Const Settings)
max_steps = experiment_settings_const["max_steps"]
n_env = experiment_settings_const["n_env"]
durations = experiment_settings_const["durations"]
enable_gcd = experiment_settings_const["enable_gcd"]

if enable_gcd:
    step_size,reduced_durations = gcd_and_reduced(durations)
else:
    step_size = 1
    reduced_durations = durations



next_sheet_row = ReadRow(EXCEL_PATH)
next_sheet_row = clean_dict_values(next_sheet_row)
next_sheet_row
print(next_sheet_row)



# Access specific parameters (Changable Settings)
data_name = next_sheet_row['Area']
n_epsiode =  int(next_sheet_row['Episodes'])
ENV_NAME = next_sheet_row['Environment Type']
REWARD_TYPE =next_sheet_row['Reward']
seed = int(next_sheet_row['Seed'])
end_time = int(next_sheet_row['Max Sumo Steps(s)'])
algorithm = next_sheet_row['Algorithm'] 

## Load Scale
precent_scale = next_sheet_row['Traffic Scale']
sumo_traffic_scale = int(10 * precent_scale)

EXPERIMENT_NAME = experiment_settings_changable["EXPERIMENT_NAME"]

if data_name=='Mosheer':
    path_data_folder =  path_main_folder + "AIST_Cleaned/data2_mosheerIsmail/"#"D:/tmp_data/"+"data2_mosheerIsmail/" # path_main_folder +"AIST_Cleaned/resco/single/"#"AIST_Cleaned/resco/single/"#"AIST_Cleaned/data3_san_stefano/"#"AIST_Cleaned/data2_mosheerIsmail/",data3_san_stefano
else:
  path_data_folder =  path_main_folder + "AIST_Cleaned/data3_san_stefano/"

path_cfg = path_data_folder +"cfg.sumocfg"

print(Fore.RED + f"CURRENT Experiment -- " + f"Data Name: {data_name} -- "+ f"Environment Type: {ENV_NAME} -- "+ f"Reward Type: {REWARD_TYPE} -- " + f"Algorithm: {algorithm}  -- "+ f"Episodes: {n_epsiode}  --  "+ f"Max Sumo Steps: {end_time}  --  "+ f"Traffic Scale: {sumo_traffic_scale}"+ f" -- Seed: {seed} " + Style.RESET_ALL)

"""## Open Sumo and Make Environment"""

conn=SumoConnection.SumoConnection(path_cfg,step_size,log_file,end_time=end_time,seed=seed)


def create_env(config_):
    env = env_classes[ENV_NAME](
        data_name=data_name,
        durations=reduced_durations,
        reward_fun=reward_func[REWARD_TYPE],
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

#from ray.tune.registry import register_env # If used later with rlib
#register_env(ENV_NAME, create_env)

set_global_conn(conn)


"""## Prepare Info (Just run once)"""

enable_editing = False
clear_file = False

if enable_editing:
    from data_parser import *
    from Utils_running_singleAgent import *
    path_info = path_project_folder+ "info_road.csv"
    agent_ids = ["1698478721"] #traci.trafficlight.getIDList(): #use full in multiagent ,now use one target agent
    agents_info = []

    for ag_i in agent_ids:
        lanes = traci.trafficlight.getControlledLanes(ag_i)
        direction_lanes = direction(traci.trafficlight.getControlledLanes(ag_i))
        out_lanes = [link[0][1] for link in traci.trafficlight.getControlledLinks(ag_i) if link]
        out_lanes = list((out_lanes))
        lanes_length = {lane: traci.lane.getLength(lane) for lane in lanes }
        edges = list(traci.lane.getEdgeID(l) for l in lanes)
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
        write_road_info(path_info,agents_info,clear_file = clear_file)
else:
    print("Skipped Writing")

#x = read_road_info(path_info, "Mosheer", match_column="data")
#x

"""## DQN

### Load DQN Parameters
"""

dqn_settings=algorithm_settings['DQN']

exploration_initial_eps = dqn_settings["exploration_initial_eps"]
exploration_final_eps = dqn_settings["exploration_final_eps"]
exploration_fraction = dqn_settings["exploration_fraction"]
learning_rate = float(dqn_settings["learning_rate"])
gamma = dqn_settings["gamma"]
policy_kwargs = dict(
    net_arch=dqn_settings["policy_kwargs"]["net_arch"],
    activation_fn=torch.nn.ReLU
)
batch_size = dqn_settings['batch_size']

"""### Prepare DQN"""

#algorithm  = 'dqn'

conn.reset()
env = create_env({})  # Create the environment instance

last_run_dict   =  ''
if algorithm == 'dqn':
    print(Fore.BLUE + f"Begin Training DQN..."+ Style.RESET_ALL)

    conn.reset()
    env = create_env({})  # Create the environment instance
    model = EpsDQN(
        RMS_DQNPolicy,
        env,
        verbose = 1,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gamma=gamma,
        exploration_initial_eps=exploration_initial_eps,  # Initial epsilon value.
        exploration_final_eps=exploration_final_eps,      # Final epsilon value.
        exploration_fraction=exploration_fraction,        # Fraction of total timesteps for linear decay.
        policy_kwargs=policy_kwargs,
        seed=seed
    )

    callback = Stable_RewardCallback(max_episodes = n_epsiode)
    time_before = time.time()
    model.learn(total_timesteps=1e9, callback=callback)
    time_after = time.time()
    rewards = callback.episode_rewards
    results_dict = env.last_run_dict # env.env for D3QN ,env for PPO

    env.close()

else:
    print("Skipped DQN")

# Save the model
# model.save("models/"+EXPERIMENT_NAME)

"""## PPO

### Hyperparameter Optimizing using Optuna (Only run once)
"""

enable_optuna_ppo = False

if enable_optuna_ppo:
    print(Fore.BLUE + f"Begin OPTUNA PPO..."+ Style.RESET_ALL)

    import optuna
    from colorama import Fore, Style
    import torch

    n_tune_episode = 10
    n_trials = 20
    seeds = [0, 1, 2]  # You can change or increase this list

    def objective(trial):
        # Sample hyperparameters
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


        avg_reward_diff = sum(rewards_diffs) / len(rewards_diffs)

        print(Fore.GREEN + f"-------------Trial {trial.number+1} finished, avg reward delta = {avg_reward_diff:.2f}-----------" + Style.RESET_ALL)
        return avg_reward_diff

    # Create Optuna study
    study = optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)

    print("Best Hyperparameters:", study.best_params)
else:
    print("Skipped Optimizing")

#study.best_params

#study.best_value

"""### Intialize Enviroment"""

env = None
vec_env = None

if algorithm == 'PPO':
    print(Fore.BLUE + f"Intializing PPO..."+ Style.RESET_ALL)

    env = create_env({})  # Create the environment instance

    vec_env = DummyVecEnv([lambda: env])

    # Wrap with VecNormalize
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
    print("Running mean:", vec_env.obs_rms.mean)
    print("Running var:", vec_env.obs_rms.var)

    # Load Settings and make model
    ppo_Settings = algorithm_settings['PPO']
    ppo_experiment_settings = ppo_Settings[str(ENV_NAME+"_"+REWARD_TYPE)]

    policy_kwargs = {
    "net_arch": [ppo_experiment_settings["net_arch"]],
    "activation_fn": torch.nn.ReLU
    }

    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=ppo_experiment_settings['learning_rate'],
        gamma=ppo_experiment_settings['gamma'],
        gae_lambda=ppo_experiment_settings['gae_lambda'],
        ent_coef=ppo_experiment_settings['ent_coef'],
        clip_range=ppo_experiment_settings['clip_range'],
        batch_size=ppo_experiment_settings['batch_size'],
        verbose=0,
        policy_kwargs=policy_kwargs,
        seed=seed
    )
    callback = Stable_RewardCallback(max_episodes = n_epsiode)
    
    print(Fore.BLUE + f"Begin Training..."+ Style.RESET_ALL)

    time_before = time.time()
    model.learn(total_timesteps=1e9, callback=callback)
    time_after = time.time()
    rewards = callback.episode_rewards
    results_dict = env.last_run_dict # env.env for D3QN ,env for PPO ##Important should call this before closing env or evaluate

    # Save the model
    #model.save("models/"+EXPERIMENT_NAME+"_test")

    """### Evaluate agent"""

    print(Fore.BLUE + f"Begin Evaluating..."+ Style.RESET_ALL)
    evaluate_results= evaluate_policy(model, env, n_eval_episodes=10, return_episode_rewards=False)[0]

"""## D3QN

### Hyperparameter Optimizing using Optuna (Only run once)
"""

enable_optuna_d3qn = False

if enable_optuna_d3qn:
    print(Fore.BLUE + f"Begin OPTUNA D3QN..."+ Style.RESET_ALL)

    import gymnasium as gym
    from gymnasium.wrappers import NormalizeObservation, NormalizeReward
    import optuna
    from colorama import Fore, Style

    env = create_env({})
    env = NormalizeObservation(env, epsilon=1e-8)

    state_size = env.observation_space.shape
    num_actions = env.action_space.n

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

        return rewards_diff


    # Create an Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    # Print best hyperparameters
    print("Best Hyperparameters:", study.best_params)
else:
  print("Skipped Optuna")

#study.best_params

#study.best_value

"""### Algorithm"""

env = None
agent = None
if algorithm == "D3QN":
    print(Fore.BLUE + f"Intializing D3QN..."+ Style.RESET_ALL)
    import gymnasium as gym
    from gymnasium.wrappers import NormalizeObservation, NormalizeReward

    # Create environment
    env = create_env({})
    env = NormalizeObservation(env, epsilon=1e-8)
    # Note in reset it will not be all zeros as normalized >> It's normal
    state_size = env.observation_space.shape
    num_actions = env.action_space.n

    # Get Paramaters
    d3qn_Settings = algorithm_settings['D3QN']
    d3qn_experiment_settings =d3qn_Settings[str(ENV_NAME+"_"+REWARD_TYPE)]
    parameters={'learning_rate': d3qn_experiment_settings['learning_rate'],
              'gamma': d3qn_experiment_settings['gamma'],
              'tau': d3qn_experiment_settings['tau'],
              'l2_reg': d3qn_experiment_settings['l2_reg'],
              'epsilon_decay': d3qn_experiment_settings['epsilon_decay'],
              'batch_size': d3qn_experiment_settings['batch_size']}

    # Make model
    agent = D3QNAgent(
            env=env,
            state_size=state_size,
            num_actions=num_actions,
            memory_size=100000,
            batch_size=parameters['batch_size'],
            gamma=parameters['gamma'],
            epsilon_start=1.0,
            epsilon_min=0.01,
            epsilon_decay=parameters['epsilon_decay'],
            learning_rate=parameters['learning_rate'],
            tau=parameters['tau'],
            update_freq=4,
            l2_reg=parameters['l2_reg'],
            random_state=seed
        )

    #path_save = 'models/tmp_unrelated.h5'
    #agent_.save_model(path_save)
    traci.simulationStep()
    #agent_.load_model(path_save)
    """### Begin Training"""


    print(Fore.BLUE + f"Begin Training D3QN..."+ Style.RESET_ALL)

    time_before=time.time()
    training_results = agent.train(
                num_episodes=n_epsiode,
                max_steps_per_episode=max_steps,
                num_points_for_average=100,
                log_interval=1)
    time_after=time.time()
    results_dict = env.env.last_run_dict # env.env for D3QN ,env for PPO ##Important should call this before closing env or evaluate
    
    print(Fore.BLUE + f"Begin Evaluating D3QN..."+ Style.RESET_ALL)
    evaluate_results=agent.evaluate(num_episodes=10)
    rewards = training_results['rewards']
    losses = training_results['losses']

# Save the trained model

#path_save = str('FINAL_'+ENV_NAME+'_'+REWARD_TYPE+'_'+EXPERIMENT_NAME)
#agent.save_model(path_save+".keras")

# Load the model

#from keras.config import enable_unsafe_deserialization
#enable_unsafe_deserialization()
#agent.load_model(path_save+".keras")

"""## Save and See results"""

time_diff = time_after - time_before
print(Fore.GREEN + f"Time taken for training: {round(time_diff,3)} seconds ({round(time_diff/60,3)} Minutes)" + Style.RESET_ALL)

last_cumulative_reward = round(rewards[-1],3)

print(Fore.MAGENTA + f"The Cumulative Reward of last Epsiode is : {last_cumulative_reward} ,Using Reward {REWARD_TYPE} " + Style.RESET_ALL)

if 'evaluate_results' in locals():
    print(Fore.CYAN + f"Avg. Reward for evaluated environment: {evaluate_results}" + Style.RESET_ALL)

for key, value in results_dict.items():
    print(Fore.CYAN + f"{key}: {round(value,3)}" + Style.RESET_ALL)
    #append_to_file("output1.txt",f"{key}: {round(value,3)}")

derivative = rewards[-1] - rewards[0]
print(Fore.GREEN + f"Derivative of reward  is {derivative}" + Style.RESET_ALL)


# Write values into that row
print(Fore.BLUE + f"Writing Info..."+ Style.RESET_ALL)

WriteRow({
    "Reward of Last Episode":last_cumulative_reward,
    "Average Reward for Evaluated Episodes":evaluate_results,
    "Derivative of Reward":derivative,
    "Waiting Time (s)":results_dict['waiting_time'],
    "Speed (m/s)":results_dict['speed'],
    "Depart Delay (s)":results_dict['depart_delay'],
    "Time Loss (s)":results_dict['time_loss'],
    "Waiting Car":results_dict['waiting_vehicles'],
    "Total Time (M)":round(time_diff/60,3),
    "Device":"colab"

},EXCEL_PATH)
print(Fore.CYAN + f"Written completed. Finished {count_full_rows(EXCEL_PATH)} Experiment"+ Style.RESET_ALL)

conn.close()
env.close()
print(Fore.CYAN + f"---------------------------------------"+ Style.RESET_ALL)
