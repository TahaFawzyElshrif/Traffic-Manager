import pandas as pd
import gymnasium as gym
import numpy as np
import traci
import warnings
from Connections.Connection import *

import math
import itertools 
from collections import deque
import random

# Suppress deprecation warnings
warnings.simplefilter("ignore", category=DeprecationWarning)


class SumoEnv(gym.Env):
    def __init__(self, traffic_light_id, count, durations, reward_fun, max_steps=50, max_sumo_steps=200,sumo_traffic_scale=1, enable_variation_action=True, config=None, seed=None):
        """
        Initializes the SUMO environment for a single agent.
        """
        super().__init__()
        
        # Set random seed
        self.seed_value = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.metadata = {"is_parallelizable": True, "render_modes": ["human"]}
        if config:
            self.horizon = config.get("horizon", 1)

        self.conn = get_global_conn()
        self.observation_size = self.conn.getLenSensors()
        self.durations = durations
        self.reward_fun = reward_fun
        self.current_step = 0
        self.done_episode = False
        
        self.agent_id = traffic_light_id
        self.max_steps = max_steps

        self.sumo_traffic_scale=sumo_traffic_scale
        self.max_sumo_steps = max_sumo_steps
        self.enable_variation_action = enable_variation_action
        
        self.python_path = ""
        self.data_path = ""
       
        self.last_run_dict = {}
        self.see_progress_each = 1

        self.last_3_signals = deque(maxlen=3)
        self.conn.set_traffic_scale(self.sumo_traffic_scale)
        # Initialize action space
        self.initialize_action_space(count)
        
        # Initialize observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_size,), dtype=np.float32
        )
        
        # Initialize state
        self.state = np.zeros(self.observation_size, dtype=np.float32)

    def initialize_action_space(self, count):
        """
        Creates the action space mapping for the traffic light agent.
        """
        space_signal = list(map("".join, itertools.product("rg", repeat=count))) if count > 0 else ["r", "g"]
        self.space = [(a, b) for a, b in itertools.product(space_signal, self.durations)]

        self.encoded_action_mapping = dict(zip(range(len(self.space)), self.space))
        self.action_space = gym.spaces.Discrete(len(self.space))

    def close(self):
        """
        Resets the environment state / clean resources
        """
        self.conn.close() 
        self._checkFinalReset()
        self.done_episode = True

        # Debug print
        print("Environment close")
    
    def reset(self, seed=None, options=None):
        """
        Resets the environment state.
        """
        # If a new seed is provided, update the seed value and set the seeds
        if seed is not None:
            self.seed_value = seed
            random.seed(seed)
            np.random.seed(seed)
        elif self.seed_value is not None:
            # Otherwise use the existing seed if one was set during initialization
            random.seed(self.seed_value)
            np.random.seed(self.seed_value)
            
        # Call the parent class reset with the seed
        super().reset(seed=seed)

        self.conn.close() 
        self._checkFinalReset()
        self.conn.initialize()
        #self.conn.reset()
        self.state = np.zeros(self.observation_size, dtype=np.float32)
        self.current_step = 0
        self.done_episode = False
        self.conn.set_traffic_scale(self.sumo_traffic_scale)

        # Debug print
        print("Environment reset")

        return self.state, {}
    
    def _checkFinalReset(self):
        last_run_dict = self.conn.get_sumo_statics(self.python_path, self.data_path)
        all_zeroes = all(value == 0 for value in last_run_dict.values())

        if all_zeroes:
            pass
        else:
            self.last_run_dict = last_run_dict
        
    def get_real_action(self, action):
        """
        Converts the action index to the actual action representation.
        """
        return self.encoded_action_mapping[action]
    
    def _all_equal(self, queue):
        return len(queue) > 0 and all(x == queue[0] for x in queue)
    
    def _get_last_two_or_default(self, dq): #before, after
        if len(dq) >= 2:
            return list(dq)[-2:]
        elif len(dq) == 1:
            return ['', dq[-1]]
        else:
            return ['', '']  # Default values if deque is empty
        
    def step(self, action):
        """
        Executes an action and advances the simulation by one step.
        """
        
        if self.done_episode:
            print("Warning: Attempting to step after episode is done.")
            return self.state, 0.0, True, True, {}
        
        # Convert gym action to SUMO action
        real_action, real_duration = self.get_real_action(action)
        self.last_3_signals.append(real_action)

        if (self.enable_variation_action):
            if (self._all_equal(self.last_3_signals)):
                real_action = self.conn.inverse_action(real_action)
            #it will append two times here, but in fact queue is full by one value so append logic won't be wrong
        self.last_3_signals.append(real_action)

        # Execute action in SUMO
        agent_steps_done = (self.conn.do_step_one_agent(
            self.agent_id, 
            real_action, 
            real_duration, 
            self.max_sumo_steps, 
            self.current_step
        ))
        
        # Get new state
        self.state = np.array(self.conn.getCurrentState(self.agent_id))
        
        # Calculate reward
        before_action, new_action = self._get_last_two_or_default(self.last_3_signals)
        reward = self.reward_fun(self.agent_id, self.state, before_action, new_action)
        
        # Check if done
        self.done_episode = (self.current_step >= self.max_steps) or agent_steps_done
        terminated = self.done_episode
        
        
        # Print progress
        if ((self.see_progress_each>0) and (((self.current_step+1)%self.see_progress_each)==0)):
            
            print(f"\rProgress: {self.current_step+1}/{min(self.max_steps,self.max_sumo_steps)}, Sumo Time {self.conn.getTime()}, state {self.state}, action {real_action}, duration {real_duration}, reward {reward}", end='', flush=True)
        
        # Increment step counter
        self.current_step += 1
        
        return self.state, reward, terminated, False, {}

    def render(self, mode='human'):
        """
        Prints the current state of the environment.
        """
        print(f"State: {self.state}, TO SEE Rendered GUI, run sumo gui")


class GroupedSumoEnv(SumoEnv):
    def __init__(self, traffic_light_id, count, durations, reward_fun, max_steps=50, max_sumo_steps=200, sumo_traffic_scale=1, enable_variation_action=True, config=None, seed=None):
        super().__init__(traffic_light_id, count, durations, reward_fun, max_steps, max_sumo_steps,sumo_traffic_scale,  enable_variation_action, config, seed)

    def initialize_action_space(self, count):
        space_signal = list(map("".join, itertools.product("rg", repeat=4)))
        self.space = [(a, b) for a, b in itertools.product(space_signal, self.durations)]
        self.encoded_action_mapping = dict(zip(range(len(self.space)), self.space))
        self.action_space = gym.spaces.Discrete(len(self.space))

    def get_lane_direction(self, lane_id):
        """Determine the primary cardinal direction of a lane in SUMO."""
        x_start, y_start = traci.lane.getShape(lane_id)[0]  # First coordinate
        x_end, y_end = traci.lane.getShape(lane_id)[-1]     # Last coordinate

        angle = math.degrees(math.atan2(y_end - y_start, x_end - x_start))

        # Optimized angle-based direction mapping
        return "E" if -45 <= angle < 45 else "N" if 45 <= angle < 135 else "W" if angle >= 135 or angle < -135 else "S"

    def get_all_lanes_action(self, action):
        """Map SUMO controlled lanes to the corresponding action signals."""
        direction_map = {"N": 0, "E": 1, "S": 2, "W": 3}
        
        controlled_lanes = traci.trafficlight.getControlledLanes(self.agent_id)
        real_action_dict = {}

        try:
            real_action_list = [
                action[direction_map[dir]] for lane in controlled_lanes if (dir := self.get_lane_direction(lane)) in direction_map
            ]
        except IndexError:
            raise ValueError(f"Invalid action index for action: {action}")

        return "".join(real_action_list), real_action_dict
    
    def get_real_action(self, action):
        action_agent_lanes, duration = self.encoded_action_mapping[action]
        real_action = self.get_all_lanes_action(action_agent_lanes)[0]
        return real_action, duration


class HighGroupedSumoEnv(SumoEnv):
    def __init__(self, traffic_light_id, count, durations, reward_fun, max_steps=50, max_sumo_steps=200,  sumo_traffic_scale=1,enable_variation_action=True, config=None, seed=None):
        super().__init__(traffic_light_id, count, durations, reward_fun, max_steps, max_sumo_steps, sumo_traffic_scale, enable_variation_action, config, seed)

    def initialize_action_space(self, count):
        space_signal = ["r"*count, "g"*count] if (count > 0) else ["r", "g"]
        self.space = [(a, b) for a, b in itertools.product(space_signal, self.durations)]

        self.encoded_action_mapping = dict(zip(range(len(self.space)), self.space))
        self.action_space = gym.spaces.Discrete(len(self.space))