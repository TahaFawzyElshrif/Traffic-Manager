import traci
from numpy import inf
import numpy as np
import gymnasium as gym
from Sensors import len_sensors
import math
from functools import reduce

def _get_traffic_lights_policies_common(n_agent, thres_count, agent_ids, shape_gym_func):
    """
    Selects traffic lights from the SUMO environment and generates policies for them.
    
    Parameters:
        n_agent (int): Number of agents to include. Stops when this number is reached.
        thres_count (int): Threshold for the number of controlled lanes per traffic light. -1 means no threshold.
        agent_ids (list): List of specific traffic light IDs to include. If provided, other parameters are ignored.
        shape_gym_func (function): A lambda/function that defines the action space shape.
    
    Returns:
        traffic_lights (list): A list of tuples with controlled lane count and traffic light ID.
        policies (dict): Dictionary mapping traffic light ID to its policy specification.
    """
    traffic_lights = []
    policies = dict({})
    tl_ids = traci.trafficlight.getIDList()  # Get all traffic light IDs from the environment

    for tl_id in tl_ids:
        controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
        n_contrlod = len(controlled_lanes)

        if (not bool(len(agent_ids)) and ((n_contrlod < thres_count) or (thres_count == -1))):
            if (len(traffic_lights) == n_agent):
                break
            traffic_lights.append((n_contrlod, tl_id))
            policies[tl_id] = (
                None,
                gym.spaces.Box(low=-inf, high=inf, shape=(len_sensors,), dtype=np.float32),
                shape_gym_func(n_contrlod),
                {}
            )
        elif bool(len(agent_ids)):
            if tl_id in agent_ids:
                traffic_lights.append((n_contrlod, tl_id))
                policies[tl_id] = (
                    None,
                    gym.spaces.Box(low=-inf, high=inf, shape=(len_sensors,), dtype=np.float32),
                    shape_gym_func(n_contrlod),
                    {}
                )  # Optional comment: Gym action space using provided shape

    return traffic_lights, policies


def get_traffic_lights_policies_full(durations, n_agent=-2, thres_count=-1, agent_ids=[]):
    """
    Generate policies using full action space: MultiDiscrete([2^x, len(durations)])
    
    Parameters:
        n_agent (int): Number of agents to include. Stops when this number is reached.
        thres_count (int): Threshold for the number of controlled lanes per traffic light. -1 means no threshold.
        agent_ids (list): List of specific traffic light IDs to include. If provided, other parameters are ignored.
        durations (list): List of durations available for signal phases.
    
    Returns:
        traffic_lights (list): A list of tuples with controlled lane count and traffic light ID.
        policies (dict): Dictionary mapping traffic light ID to its policy specification.

    """
    shape_gym_func = lambda x: gym.spaces.MultiDiscrete([2**x, len(durations)])
    return _get_traffic_lights_policies_common(n_agent, thres_count, agent_ids, shape_gym_func)

def get_traffic_lights_policies_group(durations, n_agent=-2, agent_ids=[]):
    """
    Generate group-level policies with fixed 16 signal combinations: MultiDiscrete([16, len(durations)])
    
    Parameters:
        n_agent (int): Number of agents to include. Stops when this number is reached.
        agent_ids (list): List of specific traffic light IDs to include. If provided, other parameters are ignored.
        durations (list): List of durations available for signal phases.
    
    Returns:
        traffic_lights (list): A list of tuples with controlled lane count and traffic light ID.
        policies (dict): Dictionary mapping traffic light ID to its policy specification.

    """
    shape_gym_func = lambda x: gym.spaces.MultiDiscrete([16, len(durations)])
    return _get_traffic_lights_policies_common(n_agent, -1, agent_ids, shape_gym_func)


def get_traffic_lights_policies_high_group(durations, n_agent=-2, agent_ids=[]):
    """
    Generate simplified policies with binary signal state: MultiDiscrete([2, len(durations)]) 
    
    Parameters:
        n_agent (int): Number of agents to include. Stops when this number is reached.
        agent_ids (list): List of specific traffic light IDs to include. If provided, other parameters are ignored.
        durations (list): List of durations available for signal phases.
    
    Returns:
        traffic_lights (list): A list of tuples with controlled lane count and traffic light ID.
        policies (dict): Dictionary mapping traffic light ID to its policy specification.

    """
    shape_gym_func = lambda x: gym.spaces.Discrete(2* len(durations))#gym.spaces.MultiDiscrete([2, len(durations)])
    return _get_traffic_lights_policies_common(n_agent, -1, agent_ids, shape_gym_func) #thres_count=-1 to include all traffic lights ,no need thresould


import math
from functools import reduce

def gcd_and_reduced(numbers):
    """
    Computes the greatest common divisor (GCD) of a list of numbers and returns the list of 
    numbers reduced by the GCD.

    Parameters:
    numbers (list): A list of integers for which GCD and reduced values are computed.

    Returns:
    tuple: A tuple containing:
        - gcf (int): The greatest common divisor (GCD) of the numbers.
        - reduced (list): A list of integers where each element is divided by the GCD.
    """
    # Compute the GCD of the list of numbers
    gcf = reduce(math.gcd, numbers)

    # Create a new list where each element is divided by the GCD
    reduced = [num // gcf for num in numbers]

    return gcf, reduced

