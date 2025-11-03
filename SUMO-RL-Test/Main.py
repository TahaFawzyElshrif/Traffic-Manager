#################################################################
# Intialization
#################################################################
import os
import sys
import gymnasium as gym
from stable_baselines3.dqn.dqn import DQN
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:

    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
from sumo_rl import SumoEnvironment
import numpy as np
from colorama import Fore, Style
import yaml
from stable_baselines3 import PPO, DQN
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import NormalizeObservation, NormalizeReward
import time
from stable_baselines3.common.evaluation import evaluate_policy
import optuna
import sys
import argparse

#################################################################
# Path Defination Parameters
#################################################################

# Define Parser
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

path_project_folder = parser_args.project_path
save_parameters = parser_args.save_parameters


sys.path.append(path_project_folder)
EXCEL_PATH = path_project_folder+"SUMO-RL-Test/SUMO-RL-EXP.xlsx"
PARAMETERS_PATH = path_project_folder+"SUMO-RL-Test/SUMO-RL-PARAMS.xlsx"

yaml_file = path_project_folder + "config.yaml"

# the following imports must be there as depend/has dependency of packages at path_project_folder
from data_parser import *
from Utils import *
from models.d3qn import D3QNAgent
from Callbacks import *




#################################################################
# Parameters
#################################################################
# Selective Parameters

enable_optuna = parser_args.optuna
begin_time = 10800
n_larger_step = 10800
yellow_time = 30
min_green = 17 # min duration
max_green = 90 # max duration
delta_time = 50# average duration



# Automatic Parameters

next_sheet_row = ReadRow(EXCEL_PATH) if not enable_optuna else ReadRow(PARAMETERS_PATH,"Best Parameters")
next_sheet_row = clean_dict_values(next_sheet_row)


data_name = next_sheet_row['Area'] 
n_epsiode = int(next_sheet_row['Episodes']) 
seed = int(next_sheet_row['Seed']) 
REWARD_TYPE = next_sheet_row['Reward']
algorithm = next_sheet_row['Algorithm']



n_step = int(next_sheet_row['Max Sumo Steps(s)'])

end_time = begin_time + n_step
precent_scale = .3 if enable_optuna else next_sheet_row['Traffic Scale'] 
sumo_traffic_scale = round(1+precent_scale,2)#int(10 * precent_scale)



print(Fore.RED + f"CURRENT Experiment -- Data: {data_name} | Reward: {REWARD_TYPE} | "
                 f"Algorithm: {algorithm} | Episodes: {n_epsiode} | Begin {begin_time} , End: {end_time} ({n_step} seconds) | "
                 f"Traffic Scale: {sumo_traffic_scale} | Seed: {seed}" + Style.RESET_ALL)

MODEL_NAME = f"{data_name}_{REWARD_TYPE}_{algorithm}_{sumo_traffic_scale}_{seed}"
SAVE_PATH = path_project_folder+f"/OUTPUT/SUMO_RL_{MODEL_NAME}/" if save_parameters else ""
if save_parameters:
    os.makedirs(SAVE_PATH, exist_ok=True)


# Yaml parameters
with open(yaml_file, "r") as file_:
    config = yaml.safe_load(file_)

algorithm_settings = config["algorithms_settings"]

ppo_settings = algorithm_settings[f"Sumorl_{data_name}_{REWARD_TYPE}_reward_ppo"] # assume GroupedSumoEnv most relate
d3qn_settings = algorithm_settings[f"Sumorl_{data_name}_{REWARD_TYPE}_reward_d3qn"] # assume GroupedSumoEnv most relate


experiment_settings_const = config["experiment_settings"]["const_settings"]
n_episode_evaluation = experiment_settings_const["n_episode_evaluation"]
larger_evaluation = experiment_settings_const["larger_evaluation"]



#################################################################
# Get corresponding parameters
#################################################################
if data_name == 'Mosheer':
    path_data = path_project_folder + "AIST/data2_mosheerIsmail/"
else:
    path_data = path_project_folder + "AIST/data3_san_stefano/"


if REWARD_TYPE.lower() == 'proposed':
    reward_fun = sumo_rl_proposed_reward
else:
    reward_fun = sumo_rl_literature_reward
print("USING REWARD: ",reward_fun)


#################################################################
# Other Path Parameters
#################################################################


path_net = path_data+"map.net.xml"
path_rou = path_data+"route.rou.xml"
path_stat = path_data+"osm.statistics.xml"
path_trip = path_data+"tripinfo.xml" 
log_dir = path_project_folder+"log_dir/"

#################################################################
# Train
#################################################################
import os
import sys
import traci
from io import StringIO

# Temporarily suppress SUMO warnings
#stderr_backup = sys.stderr
#sys.stderr = open(os.devnull, 'w') 


parameters = {
    "net_file": path_net,
    "route_file": path_rou,
    "single_agent": True,
    "use_gui": False,
    "delta_time": delta_time,
    "min_green":min_green,
    "max_green":max_green,
    "yellow_time": yellow_time,
    "begin_time":begin_time ,
    "num_seconds": n_step,
    "reward_fn": reward_fun,
    "sumo_seed":seed,
    "observation_class":SumoRL_State_Wrapper,
    "max_depart_delay":300,
    "time_to_teleport":1000,

    "additional_sumo_cmd": f"--no-warnings -e {begin_time+n_step} --statistic-output {path_stat} --tripinfo-output {path_trip} --scale {sumo_traffic_scale} --step-length {1} --collision.action warn --collision.check-junctions True --collision.mingap-factor 0.1 --pedestrian.striping.mingap-to-vehicle 0.25 --weights.random-factor 1.5 --threads 1 --log sumo.log"
}






#################################################################
# PPO
#################################################################

if algorithm == 'PPO':
  # ----------------------------
  # Optuna
  # ----------------------------
  def make_env():
       return SumoEnvironment(**parameters)
  
  if enable_optuna:
      print(Fore.BLUE + "Begin OPTUNA PPO..." + Style.RESET_ALL)
      

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
              vec_env = DummyVecEnv([make_env])
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
              vec_env.close()
              
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
        },PARAMETERS_PATH,"Best Parameters")

      print(Fore.CYAN + f"Written completed. " + Style.RESET_ALL)
      
  
  else:
      print("Skipped Optimizing PPO")
      

      print(Fore.BLUE + "Initializing PPO..." + Style.RESET_ALL)
      

      policy_kwargs = {"net_arch": [ppo_settings["net_arch"]], "activation_fn": torch.nn.ReLU}

      vec_env = DummyVecEnv([make_env])
      vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)


      vec_env.reset()
      vec_env.reset()
      
      
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
      model.learn(total_timesteps=10, callback=callback)
      time_after = time.time()
      

      rewards = callback.episode_rewards
      last_cumulative_reward = round(rewards[-1], 3)
      

      vec_env.save(os.path.join(SAVE_PATH, "vecnormalize.pkl"))
      vec_env.close()

      
      # Set the max sumo time to max possible time used in evaluation + additional time ,this to prevent sumo automatic reset after end which clear osm file
      # additional things are made to prevent this situation , as looping with large n time but stop when reaching max wanted time

      
      parameters['num_seconds'] = larger_evaluation+1000
      parameters["additional_sumo_cmd"]= f"--no-warnings -e {begin_time+larger_evaluation+1000} --statistic-output {path_stat} --tripinfo-output {path_trip} --scale {sumo_traffic_scale} --step-length {1} --collision.action warn --collision.check-junctions True --collision.mingap-factor 0.1 --pedestrian.striping.mingap-to-vehicle 0.25 --weights.random-factor 1.5 --threads 1 --log sumo.log"

      print(Fore.BLUE + f"Begin Evaluating PPO On Same Time..."+ Style.RESET_ALL)

      rewards_evaluated_envs = []

      results_dict = {}
      for j in range(n_episode_evaluation):
            eval_env = SumoEnvironment(**parameters)
            vec_eval_env = DummyVecEnv([lambda: eval_env])
            vec_eval_env = VecNormalize(vec_eval_env, norm_obs=True, norm_reward=False)
            vec_eval_env = VecNormalize.load(os.path.join(SAVE_PATH, "vecnormalize.pkl"), vec_eval_env)

            obs = vec_eval_env.reset()
            c_reward = 0


            for i in range(10000000):
                              # Not using 'done' here because yellow_time and delta_time make it hard to control the exact end time.
                              # This loop is only for benchmarking purposes, so it's fine.
                              # It may cause 'tcpip::Socket::recvAndCheck @ recv: peer shutdown' when SUMO closes automatically.

                              action, _ = model.predict(obs, deterministic=True)
                              obs, reward, done, _ = vec_eval_env.step(action)
                              c_reward += reward
                              print(traci.simulation.getTime(),"of",(begin_time + n_step)) # additional print to prevent stuckking

                              if done:
                                break
                              if ( traci.simulation.getTime() >= (begin_time + n_step)):
                                break

            eval_env.close()
            results_dict = get_sumo_statics(path_data)
            rewards_evaluated_envs.append(c_reward)
            
      print(f"  {results_dict}")
      avg_reward_evaluated = sum(rewards_evaluated_envs)/len(rewards_evaluated_envs)

      

      print(Fore.BLUE + f"Begin Evaluating PPO For Larger Time {larger_evaluation/60} M..."+ Style.RESET_ALL)
 
      eval_env = SumoEnvironment(**parameters)
      vec_eval_env = DummyVecEnv([lambda: eval_env])
      vec_eval_env = VecNormalize(vec_eval_env, norm_obs=True, norm_reward=False)
      vec_eval_env = VecNormalize.load(os.path.join(SAVE_PATH, "vecnormalize.pkl"), vec_eval_env)

      obs = vec_eval_env.reset()
      c_reward = 0


      for i in range(10000000):
                        # Not using 'done' here because yellow_time and delta_time make it hard to control the exact end time.
                        # This loop is only for benchmarking purposes, so it's fine.
                        # It may cause 'tcpip::Socket::recvAndCheck @ recv: peer shutdown' when SUMO closes automatically.

                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, done, _ = vec_eval_env.step(action)
                        c_reward += reward
                        print(traci.simulation.getTime(),"of",(begin_time + larger_evaluation)) # additional print to prevent stuckking

                        if done:
                          break
                        if ( traci.simulation.getTime() >= (begin_time + larger_evaluation)):
                          break

      eval_env.close()
      #vec_eval_env.close()
      results_dict_large_eval = get_sumo_statics(path_data)
      print(f"Eval METRICES {results_dict_large_eval}")

      if save_parameters:
            model.save(os.path.join(SAVE_PATH, "ppo_model"))
            print(f"MODEL IS SAVED AT {os.path.join(SAVE_PATH, "ppo_model")}")

      else:
            os.remove(os.path.join(SAVE_PATH, "vecnormalize.pkl"))
            print(f"REMOVED TEMP FILE  {os.path.join(SAVE_PATH, "vecnormalize.pkl")}")


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
                env = SumoEnvironment(**parameters)
                env = NormalizeObservation(env, epsilon=1e-8)

                state_size = env.observation_space.shape
                num_actions = env.action_space.n

                env.reset()
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
                env.close()
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
        },PARAMETERS_PATH,"Best Parameters")

        print(Fore.CYAN + f"Written completed. " + Style.RESET_ALL)
        
    else:
        print("Skipped Optuna")

        print(Fore.BLUE + "Initializing D3QN..." + Style.RESET_ALL)
        
        env = SumoEnvironment(**parameters)

        

        env = NormalizeObservation(env, epsilon=1e-8)
        state_size = env.observation_space.shape
        num_actions = env.action_space.n

        d3qn_settings = algorithm_settings['D3QN'][f"{data_name}_{"GroupedSumoEnv"}_{REWARD_TYPE}_reward"]
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
            max_steps_per_episode=100000,
            num_points_for_average=100,
            log_interval=1
        )
        time_after = time.time()
        rewards = training_results['rewards']

        # Evaluation for same time
        
        print(Fore.BLUE + f"Begin Evaluating D3QN On Same Time..."+ Style.RESET_ALL)
        rewards_evaluated_envs = []

        for i in range(n_episode_evaluation):
                env_eval = SumoEnvironment(**parameters)
                env_eval = NormalizeObservation(env_eval, epsilon=1e-8)


                obs,info = env_eval.reset()
                c_reward = 0


                while not (traci.simulation.getTime() >= (begin_time + n_step)): 
                                    # Not using 'done' here because yellow_time and delta_time make it hard to control the exact end time.
                                    # This loop is only for benchmarking purposes, so it's fine.
                                    # It may cause 'tcpip::Socket::recvAndCheck @ recv: peer shutdown' when SUMO closes automatically.

                                    action = agent.get_action(obs)
                                    obs, reward, done, _, info = env_eval.step(action)
                                    c_reward += reward
                                    if done:
                                        break
                env_eval.close()
                rewards_evaluated_envs.append(c_reward)


        avg_reward_evaluated = sum(rewards_evaluated_envs)/len(rewards_evaluated_envs)  
        results_dict = get_sumo_statics(path_data)
        print(results_dict)
        
        # Evaluation for larger time
        
       
        print(Fore.BLUE + f"Begin Evaluating D3QN For Larger Time {larger_evaluation/60} M..."+ Style.RESET_ALL)

        

        parameters['num_seconds'] = larger_evaluation
        parameters["additional_sumo_cmd"]= f"--no-warnings -e {begin_time+larger_evaluation} --statistic-output {path_stat} --tripinfo-output {path_trip} --scale {sumo_traffic_scale} --step-length {1} --collision.action warn --collision.check-junctions True --collision.mingap-factor 0.1 --pedestrian.striping.mingap-to-vehicle 0.25 --weights.random-factor 1.5 --threads 1 --log sumo.log"

        env_eval = SumoEnvironment(**parameters)
        env_eval = NormalizeObservation(env_eval, epsilon=1e-8)


        obs,info = env_eval.reset()
        c_reward = 0


        while not (traci.simulation.getTime() >= (begin_time + larger_evaluation)): 
                            # Not using 'done' here because yellow_time and delta_time make it hard to control the exact end time.
                            # This loop is only for benchmarking purposes, so it's fine.
                            # It may cause 'tcpip::Socket::recvAndCheck @ recv: peer shutdown' when SUMO closes automatically.

                            action = agent.get_action(obs)
                            obs, reward, done, _, info = env_eval.step(action)
                            c_reward += reward
                            if done:
                               break
        env_eval.close()

        results_dict_large_eval = get_sumo_statics(path_data)
        print(results_dict_large_eval)
        if save_parameters:
            agent.save_model(os.path.join(SAVE_PATH, "d3qn_model.keras"))
            print(f"MODEL IS SAVED AT {os.path.join(SAVE_PATH, "d3qn_model")}")


####################################################
## Write
####################################################
if enable_optuna:
    pass
else:
    time_diff = time_after - time_before
    last_cumulative_reward = round(rewards[-1], 3)
    derivative = rewards[-1] - rewards[0]

    if save_parameters:
        import matplotlib.pyplot as plt 
        from datetime import datetime
        
        now = datetime.now()
        
        plt.plot(rewards)
        plt.savefig(SAVE_PATH+f"reward_during_training.png")   # Save the figure

    
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

        "Average Reward for Evaluated Episodes With Same Time":avg_reward_evaluated,
        "Waiting Car on Final Test":results_dict_large_eval['waiting_vehicles'],
        "Time Loss (s) on Final Test":results_dict_large_eval['time_loss'],
        "Depart Delay (s) on Final Test": results_dict_large_eval['depart_delay'],
        "Speed (m/s) on Final Test":results_dict_large_eval['speed'],
        "Waiting Time (s) on Final Test":results_dict_large_eval['waiting_time'],
        "Reward on Final Test": c_reward,

        "Total Time Of Training (M)": round(time_diff/60, 3),
        "Device": "colab"
    }, EXCEL_PATH)
    
    print(Fore.CYAN + f"Written completed. Finished {count_full_rows(EXCEL_PATH)} Experiments" + Style.RESET_ALL)

    env.close()
    print(Fore.CYAN + "---------------------------------------" + Style.RESET_ALL)
