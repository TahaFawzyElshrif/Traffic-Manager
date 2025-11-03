import numpy as np
from Connections.Connection import get_global_conn

# Helper function to calculate the Hamming distance between two states
def _hamming(state1, state2):
    """
    Calculates the Hamming distance between two states.
    
    Parameters:
    state1 (tuple): The first state to compare.
    state2 (tuple): The second state to compare.
    
    Returns:
    int: The Hamming distance, i.e., the number of differing elements between the two states.
    """
    return sum(map(str.__ne__, state1, state2)) 

# Reward function based on proposed traffic state metrics
def reward_proposed(agent, single_state, last_action, action):
    """
    Calculates the independent reward for a single traffic light based on various traffic metrics.
    
    Parameters:
    agent (String): The agent id.
    single_state (tuple): Observation Class has getStataSpace():
    Contains traffic state metrics in the following order:
        avg_speed (float) - Average vehicle speed.
        var_speed (float) - Variance of vehicle speeds.
        avg_waiting_time (float) - Average waiting time of vehicles.
        var_waiting_time (float) - Variance of waiting times.
        avg_throughput (float) - Number of vehicles passing.
        avg_queue_length (float) - Average queue length.
        avg_Occupancy (float) - Road occupancy percentage.
    last_action: Previous action, not used in the calculation.
    action: Current action, not used in the calculation.
    
    Returns:
    float: Our Proposed reward for the given traffic state.
    """
    MAX_THROUGHPUT = 50.0
    MAX_QUEUE = 30.0
    MAX_SPEED = 60.0
    MAX_WAITING = 120.0

    w1 = 0.6
    w2 = 0.4
    
    scale_speed=1
    scale_waiting=1
    scale_efficiency =1
    eta = 1e-6  # To prevent division by zero

    avg_speed, var_speed, avg_waiting_time, var_waiting_time, avg_throughput, avg_queue_length, avg_Occupancy = np.array(single_state.get_state_space())

    # Speed reward
    speed_norm = avg_speed / MAX_SPEED  # 0 to 1
    variance_penalty = 1 / (1 + var_speed)  # 1 if var=0, decreases with variance
    speed_term = speed_norm * variance_penalty


    # Waiting time penalty
    waiting_norm = avg_waiting_time / MAX_WAITING  # 0 to 1
    variance_penalty_wait = 1 / (1 + var_waiting_time)
    waiting_term = waiting_norm * variance_penalty_wait
    
    # Traffic efficiency metric
    traffic_efficiency = (w1/MAX_THROUGHPUT * avg_throughput) - (w2/MAX_QUEUE * avg_queue_length)

    # Final reward formula
    independent_part_reward = scale_speed * speed_term - (scale_waiting * waiting_term) + (scale_efficiency * traffic_efficiency)

    return independent_part_reward

# Reward function based on literature data from the road network
def reward_liter(agent, single_state, last_action, action):
    """
    Calculates the reward based on road literature data.
    
    Parameters:
    agent (String): The agent id.
    single_state: State information, not used in this calculation.
    last_action: Previous action, not used in the calculation.
    action: Current action, not used in the calculation.
    
    Returns:
    float: The literature based reward for the given traffic state ,Used to compare results.
    """
    conn = get_global_conn()
    road_inf = single_state.get_waiting_time()  #conn.get_detailed_road_literature(agent)
    if road_inf is None or len(road_inf) == 0:  # This covers None or an empty list
        return 0.0  # Fallback if no road information is available

    lambda_ = 0.15
    Tw = 20
    alpha = 2

    terms = []
    for wi in road_inf:
        if wi is not None:  # Check for valid data
            val = lambda_ * (1 - ((wi / Tw) ** alpha))
            terms.append(val)

    if not terms:
        return 0.0  # Return 0 if no valid terms are calculated

    return np.mean(terms)

# Reward function based on projected road metrics
def reward_proj(agent, single_state, last_action, action):
    """
    Calculates the reward based on projected road data.
    
    Parameters:
    agent (String): The agent id.
    single_state: State information, not used in this calculation.
    last_action: Previous action to calculate reward difference.
    action: Current action to compare with the last action.
    
    Returns:
    float: The Project reward for the given traffic state ,Used for baseline.
    """
    conn = get_global_conn()
    observation = conn.get_improved_road_proj(agent)

    reward = 0
    occupancy = observation[1]
    haltingCars = observation[2]
    emergencyStops = observation[4]

    trafficFlow = occupancy / haltingCars if haltingCars > 0 else occupancy

    if last_action is None:
        return 0  # No reward if last action is None
    
    # Calculate reward based on traffic flow and penalties
    reward = reward + trafficFlow - _hamming(last_action, action) - emergencyStops

    return reward
