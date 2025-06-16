from Sensors import getSumoSensors_full ,len_sensors ,len_optimized_sensors
import traci
import os
import json
from dotenv import load_dotenv
import uuid
import random
import numpy as np



global_conn = None



def set_global_conn(conn):
  global global_conn
  global_conn = conn

def get_global_conn():
  return global_conn


class Connection:
    """
    Base class for managing traffic simulation connections.
    """

    def __init__(self):
        """
        Initializes metrics for traffic simulation.
        """
        self.total_waiting_time = 0
        self.total_speed = 0
        self.total_time_loss = 0
        self.total_depart_delay = 0
        self.vehicle_count = 0
        self.departed_count = 0

    def initialize(self):
        pass

    def set_traffic_scale(self,scale):
        pass

    def get_improved_road_proj(self, agent):
        return np.array([0]*5)
    
    def get_sumo_statics(self,python_path, data_path):
        return {}
    def getTime(self):
        return 0

    def inverse_action(self, action):
        """
        Inverts a traffic light action string.
        """
        return ''.join(['g' if x == 'r' else 'r' for x in action])

    def close(self):
        """
        Closes the connection.
        """
        pass

    def reset(self):
        """
        Resets the simulation.
        """
        pass

    def done_cond(self):
        """
        Checks if the simulation is done.
        """
        pass

    def do_steps_duration(self, duration, max_sumo_step, agent, traffic_scale):
        """
        Performs simulation steps for a given duration.
        """
        pass

    def do_step_one_agent(self, agent, new_action, duration, max_sumo_step, traffic_scale):
        """
        Performs a simulation step for a single agent.
        """
        pass

    def getCurrentState(self, agent):
        """
        Retrieves the current state for a given agent.
        """
        pass

    def getLenSensors(self):
        """
        Retrieves the number of sensors.
        """
        return 7

    def get_detailed_road_literature(self, agent):
        """
        Retrieves detailed road literature data for an agent.
        """
        pass

    def get_detailed_road_proj(self, agent):
        """
        Retrieves detailed road projection data for an agent.
        """
        return np.array([0]*5)

    def set_traffic_scale(self ,scale):
        """
        Set the Scale of traffic for simulation steps
        """
        pass

    def reset_metrices(self):
        """
        Reset Metrices ,used to make final metrices only for last learned episode
        """
        self.total_waiting_time = 0
        self.total_speed = 0
        self.total_time_loss = 0
        self.total_depart_delay = 0
        self.vehicle_count = 0
        self.departed_count = 0