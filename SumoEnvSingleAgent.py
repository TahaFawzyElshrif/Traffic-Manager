import math
import itertools
import random
import warnings
from collections import deque

import gymnasium as gym
import numpy as np
import pandas as pd
import traci

from Connections.Connection import get_global_conn
from Observations.sumo_obs import DefaultObservation
from data_parser import read_road_info

# Suppress deprecation warnings
warnings.simplefilter("ignore", category=DeprecationWarning)


class Agent:
    """
    Represents a single traffic light agent in the SUMO environment.

    Attributes:
        env (SumoEnv): The environment instance the agent belongs to.
        agent_id (str): The unique ID of the traffic light.
        min_phase (int): Minimum green phase duration.
        max_phase (int): Maximum green phase duration.
        next_action_time (float): Time step when the next action is allowed.
        fixed_ts (bool): Whether the traffic signal is fixed-timing.
        is_yellow (bool): Whether the signal is currently yellow.
        time_since_last_phase_change (int): Steps since the last phase change.
        current_phase (int): The current traffic light phase.
    """

    def __init__(self, env, agent_id):
        self.env = env
        self.agent_id = agent_id
        self.min_phase = min(self.env.durations)
        self.max_phase = max(self.env.durations)
        self.step_size = self.env.step_size

        self.next_action_time = self.env.begin
        self.fixed_ts = False
        self.is_yellow = False
        self.time_since_last_phase_change = 0
        self.current_phase = 0

    @property
    def time_to_act(self):
        """Check if it's time for the agent to take the next action."""
        return (self.next_action_time == self.env.conn.getTime())

    def set_next_phase(self, new_phase: int):
        """
        Change the traffic light to a new phase.

        Args:
            new_phase (int): The phase index to set.
        """
        new_action, new_duration = self.env.get_real_action(new_phase)
        current_action = self.env.get_real_action(self.current_phase)[0]
        

        # Case 1: Same phase - just continue with possibly new duration
        if new_phase == self.current_phase:
            
            # If we've been in this phase too long, we might want to force a brief transition
            # But for now, just continue the phase
            self.env.conn.do_step_one_agent(self.agent_id, current_action)
            
            # Use the new duration for timing
            self.next_action_time = self.env.conn.getTime() + new_duration * self.step_size
        
        # Case 2: Different phase but minimum time not met - continue current phase
        elif self.time_since_last_phase_change < self.min_phase:
            
            self.env.conn.do_step_one_agent(self.agent_id, current_action)
            
            # Keep current phase timing
            current_duration = self.env.get_real_action(self.current_phase)[1]
            self.next_action_time = self.env.conn.getTime() + current_duration * self.step_size
        
        # Case 3: Different phase and can transition - apply yellow first
        else:
            
            yellow_action = self.env.corresponding_yellow[current_action, new_action]
            
            self.env.conn.do_step_one_agent(self.agent_id, yellow_action)
            
            # Update state for yellow phase
            self.current_phase = new_phase
            self.is_yellow = True
            self.time_since_last_phase_change = 0
            
            # Next action time includes yellow duration
            self.next_action_time = (self.env.conn.getTime() + 
                                  self.env.yellow_time + 
                                  new_duration * self.step_size)
        
    
    def update(self):
        """Advance the agent's internal timer and change yellow to green if needed."""
        self.time_since_last_phase_change += 1
        if self.is_yellow and self.time_since_last_phase_change == self.env.yellow_time:
            self.env.conn.do_step_one_agent(
                self.agent_id,
                self.env.get_real_action(self.current_phase)[0],
                
            )
            self.is_yellow = False


class SumoEnv(gym.Env):
    """
    SUMO single-agent traffic light control environment.
    """

    def __init__(self, data_name, durations, reward_fun,step_size=1,
                 obs_class=DefaultObservation, path_info="info_road.csv",
                 yellow_time=7, max_steps=50, 
                 sumo_traffic_scale=1, enable_variation_action=True,
                 config=None, seed=None):
        super().__init__()

        self.yellow_time = yellow_time
        self.seed_value = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.metadata = {"is_parallelizable": True, "render_modes": ["human"]}
        if config:
            self.horizon = config.get("horizon", 1)

        # SUMO connection
        self.conn = get_global_conn()
        self.begin = self.conn.getTime()
        self.observation_size = self.conn.getLenSensors()
        self.durations = durations
        self.reward_fun = reward_fun
        self.current_step = 0
        self.done_episode = False
        self.fixed_ts = False
        self.max_steps = max_steps
        self.step_size = step_size
        self.sumo_traffic_scale = sumo_traffic_scale
        self.enable_variation_action = enable_variation_action
        self.delta_time = 1  # Default simulation step duration

        self.python_path = ""
        self.data_path = ""
        self.last_run_dict = {}
        self.see_progress_each = 1

        self.conn.set_traffic_scale(self.sumo_traffic_scale)

        # Observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.observation_size,), dtype=np.float32
        )
        self.state = np.zeros(self.observation_size, dtype=np.float32)

        # Load agent info
        agent_info = read_road_info(path_info, data_name, "data")
        self.agent_info = {agent['agent_id']: agent for agent in agent_info}
        self.agent_id = str(list(self.agent_info.keys())[0])
        self.agents = {self.agent_id: Agent(self, self.agent_id)}

        self.count_controlled = len(self.agent_info[int(self.agent_id)]['lanes']) # int is default type at agent info  so reset in some places to prevent error
        self.initialize_action_space(self.count_controlled)

        # Observations
        self.obs_class = obs_class
        self.observations = {
            self.agent_id: self.obs_class(
                self,
                self.agent_id,
                self.agent_info[int(self.agent_id)]['lanes'],
                self.agent_info[int(self.agent_id)]['edge']
            )
        }

    def getCorrespondingYellow(self, a, b):
        """Return a yellow phase string between two given phases."""
        return ''.join(['y' if a[i] != b[i] else a[i] for i in range(len(a))])

    def initialize_action_space(self, count):
        """Create the action space mapping for the agent."""
        space_signal = list(map("".join, itertools.product("rg", repeat=count))) if count > 0 else ["r", "g"]
        self.space = [(a, b) for a, b in itertools.product(space_signal, self.durations)]
        self.corresponding_yellow = {
            (seg1, seg2): self.getCorrespondingYellow(seg1, seg2)
            for seg2 in space_signal for seg1 in space_signal
        }
        self.encoded_action_mapping = dict(zip(range(len(self.space)), self.space))
        self.action_space = gym.spaces.Discrete(len(self.space))

    def close(self):
        """Close the environment and release resources."""
        self.conn.close()
        self._checkFinalReset()
        self.done_episode = True
        print("Environment closed.")

    def reset(self, seed=None, options=None):
        """Reset the environment state."""
        if seed is not None:
            self.seed_value = seed
            random.seed(seed)
            np.random.seed(seed)
        elif self.seed_value is not None:
            random.seed(self.seed_value)
            np.random.seed(self.seed_value)

        super().reset(seed=seed)

        self.conn.close()
        self._checkFinalReset()
        self.conn.initialize()
        self.state = np.zeros(self.observation_size, dtype=np.float32)
        self.current_step = 0
        self.done_episode = False
        self.conn.set_traffic_scale(self.sumo_traffic_scale)

        # FIX: Properly reset agents after SUMO reconnection
        self.begin = self.conn.getTime()  # Update begin time
        
        # Reset each agent to match new SUMO state
        for agent_id, agent in self.agents.items():
            agent.next_action_time = self.begin  # Allow immediate action
            agent.time_since_last_phase_change = 0
            agent.is_yellow = False
            agent.current_phase = 0  # Reset to initial phase
            
        print("Environment reset - agents properly initialized.")
        return self.state, {}

    def _checkFinalReset(self):
        """Check SUMO run statistics after reset."""
        last_run_dict = self.conn.get_sumo_statics(self.data_path)
        if last_run_dict is None:
            print("⚠ Skipping episode: statistics file is corrupted or incomplete")
        else:
            self.last_run_dict = last_run_dict

    def get_real_action(self, action):
        """Convert an action index to the actual SUMO signal and duration."""
        return self.encoded_action_mapping[action]

    def _run_steps(self):
        """Advance the simulation until it's time to act."""
        time_to_act = False
        iteration = 0
        while not time_to_act:
            self.conn.step()
            iteration += 1
            #print("------sub iteration ",iteration)
            for ts in self.agents.values():
                ts.update()
                if ts.time_to_act:
                    time_to_act = True
            if iteration > 100:
                print("Breaking due to too many iterations")
                break
            if self.conn.done:
                break

    def _apply_actions(self, actions):
        """Apply the given actions to agents."""
        if len(self.agents) == 1:
            agent_obj = list(self.agents.values())[0]
            if agent_obj.time_to_act:
                agent_obj.set_next_phase(actions)

    def step(self, action):
        """Advance the simulation by one step given an action."""
        if self.done_episode:
            print("Warning: Step called after episode ended.")
            return self.state, 0.0, True, True, {}

        if self.fixed_ts or action is None or action == {}:
            for _ in range(self.delta_time):
                self.conn.step()
        else:
            self._apply_actions(action)
            self._run_steps()

        current_observ = self.observations[self.agent_id]
        self.state = np.array(current_observ.get_state_space())
        reward = self.reward_fun(self.agent_id, current_observ, None, None)

        self.done_episode = (self.current_step >= self.max_steps) or (self.conn.done)
        terminated = self.done_episode

        if self.see_progress_each > 0 and (self.current_step + 1) % self.see_progress_each == 0:
            print(f"\rProgress: {self.current_step+1}/{self.max_steps}, "
                  f"Sumo Time {self.conn.getTime()}, state {self.state}, reward {reward}", end='', flush=True)

        self.current_step += 1
        return self.state, reward, terminated, False, {}

    def render(self, mode='human'):
        """Print the current state (GUI rendering is SUMO's responsibility)."""
        print(f"State: {self.state}, TO SEE Rendered GUI, run SUMO GUI.")


class GroupedSumoEnv(SumoEnv):
    """SUMO environment for grouped traffic light control."""
    def __init__(self, data_name, durations, reward_fun,step_size=1,
                 obs_class=DefaultObservation, path_info="info_road.csv",
                 yellow_time=7, max_steps=50, 
                 sumo_traffic_scale=1, enable_variation_action=True,
                 config=None, seed=None):
                 
        super().__init__(data_name, durations, reward_fun,step_size,
                 obs_class, path_info,
                 yellow_time, max_steps,
                 sumo_traffic_scale, enable_variation_action,
                 config, seed)

        self.agent_directions = self.agent_info[int(self.agent_id)]['direction_lanes']

                         

    def initialize_action_space(self, count):
        space_signal = list(map("".join, itertools.product("rg", repeat=4)))
        self.corresponding_yellow = {
            (seg1, seg2): self.getCorrespondingYellow(seg1, seg2)
            for seg2 in space_signal for seg1 in space_signal
        }
        self.space = [(a, b) for a, b in itertools.product(space_signal, self.durations)]
        self.encoded_action_mapping = dict(zip(range(len(self.space)), self.space))
        self.action_space = gym.spaces.Discrete(len(self.space))
        self.direction_map = {"N": 0, "E": 1, "S": 2, "W": 3}

    def get_all_lanes_action(self, action):
        """Map SUMO controlled lanes to the corresponding action signals."""
        return "".join([action[self.direction_map[dir_]] for dir_ in self.agent_directions])

    def get_real_action(self, action):
        """Convert action index to real grouped lane action."""
        action_agent_lanes, duration = self.encoded_action_mapping[action]
        real_action = self.get_all_lanes_action(action_agent_lanes)
        return real_action, duration


class HighGroupedSumoEnv(SumoEnv):
    """SUMO environment with simplified two-phase signals."""
    def initialize_action_space(self, count):
        space_signal = ["r"*count, "g"*count] if count > 0 else ["r", "g"]
        self.corresponding_yellow = {
            (seg1, seg2): self.getCorrespondingYellow(seg1, seg2)
            for seg2 in space_signal for seg1 in space_signal
        }
        self.space = [(a, b) for a, b in itertools.product(space_signal, self.durations)]
        self.encoded_action_mapping = dict(zip(range(len(self.space)), self.space))
        self.action_space = gym.spaces.Discrete(len(self.space))
