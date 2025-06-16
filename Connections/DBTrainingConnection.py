from Sensors import getSumoSensors_full ,len_sensors ,len_optimized_sensors
import traci
import os
import json
from dotenv import load_dotenv
import uuid
import random
import numpy as np
from Connections.Connection import Connection

class DBTrainingConnection(Connection):
    """
    Connection used for online reinforcement learning from a database.
    """

    def __init__(self, database_link):
        """
        Initializes the database training connection.
        """
        self.state = dict({})

    def close(self):
        """
        Closes the database training connection.
        """
        pass

    def reset(self):
        """
        Resets the database training connection.
        """
        pass

    def done_cond(self):
        """
        Checks if the database simulation is done.
        """
        pass

    def do_steps_duration(self, duration, max_sumo_step, agent, traffic_scale):
        """
        Performs simulation steps for a given duration in the database training connection.
        """
        pass

    def do_step_one_agent(self, agent, new_action, duration, max_sumo_step, traffic_scale):
        """
        Performs a simulation step for a single agent in the database training connection.
        """
        pass

    def getCurrentState(self, agent):
        """
        Retrieves the current state for a given agent.
        """
        return self.state[agent]

    def setCurrentState(self, agent, state):
        """
        Sets the current state for a given agent.
        """
        self.state[agent] = state

    def getLenSensors(self):
        """
        Retrieves the number of sensors.
        """
        return len_sensors  # or len_optimized_sensors
