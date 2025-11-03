from Connections.Connection import Connection

import gymnasium as gym
import numpy as np

class ONLINEDBConn(Connection):
    def __init__(self,n_state,step_size=1):
        
        
        self.total_states = n_state
        self.step_size = step_size
        self.state_id = 0

        self.time = 0 
        self.done = False
        self.current_action = 0


    def do_step_one_agent(self,agent, action,action_id): # should only set ,step do action (single time)
        self.state_id = (self.state_id + 1) % self.total_states 
        self.current_action = action_id
 
        
        
    def step(self):
        self.time+=self.step_size
        self.done = self.state_id >= self.total_states

       
    def getTime(self):
        return self.time
        

    def reset(self, seed=None, options=None):
        self.done = False
        self.state_id = 0
        self.current_action = 0
        self.time = 0
        return np.zeros(7, dtype=np.float32), {}

    def render(self, mode="human"):
        print(f"Current State ID: {self.current_id}")

    def close(self):
        pass

    def set_traffic_scale(self,scale):
        pass
