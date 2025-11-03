
import numpy as np
import traci


def hamming(state1, state2):
  return sum(map(str.__ne__, state1, state2))

def firstRewardaFunction(agent,action, observation,last_action):
    try:
        reward = 0
        waitingTime = observation[0]

        if waitingTime == 0:
            reward = reward + 1
        elif (waitingTime / 10) < 0.2:
            reward = reward - 0.5
        elif (waitingTime / 10) > 0.5:
            reward = reward - 1

        reward = reward + 0.1 * action.count("g") - 0.1 * action.count("r")

        return reward
    except:
        return 0

def secondRewardFunction(agent,action, observation, last_action):
    try:
        reward = 0
        occupancy = observation[1]
        haltingCars = observation[2]
        emergencyStops = observation[4]

        trafficFlow = occupancy / haltingCars if haltingCars else occupancy

        if (last_action is None):
            return 0
        
        reward = reward + trafficFlow - hamming(last_action, action) - emergencyStops

        return  reward
    except:
        return 0
    

def getStateProposed(tl_id):
    """
    Retrieves various traffic-related metrics from SUMO for a given traffic light ID.
    
    This function collects data such as vehicle speeds, waiting times, queue lengths, 
    throughput, and occupancy for all lanes controlled by the specified traffic light.

    Parameters:
        tl_id (str): The ID of the traffic light in SUMO.

    Returns:
        tuple: A tuple containing the following metrics:
            - avg_speed (float): Average speed of vehicles.
            - var_speed (float): Variance in vehicle speeds.
            - avg_waiting_time (float): Average waiting time of vehicles.
            - var_waiting_time (float): Variance in waiting time.
            - avg_throughput (float): Average number of vehicles passing per step.
            - avg_queue_length (float): Average number of halted vehicles.
            - avg_Occupancy (float): Average lane occupancy percentage.
    """

    vehicle_waiting = []  # Stores waiting times of vehicles
    vehicle_speeds = []  # Stores speeds of vehicles
    edges = []  # Stores unique edge IDs corresponding to lanes
    throughputs = []  # Stores number of vehicles passing per step
    queue_lengths = []  # Stores number of halted vehicles per edge
    Occupancies = []  # Stores lane occupancy percentages
    
    # Initialize output variables
    avg_waiting_time = 0
    var_waiting_time = 0
    avg_speed = 0
    var_speed = 0
    avg_throughput = 0
    avg_queue_length = 0
    avg_Occupancy = 0  # Percentage of occupied space in lanes

    # Get all lanes controlled by the traffic light
    controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)

    for lane in controlled_lanes:
        # Get vehicle IDs currently in the lane
        vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)

        # Collect speed and waiting time of vehicles in the lane
        vehicle_speeds += [traci.vehicle.getSpeed(vehicle_id) for vehicle_id in vehicle_ids]
        vehicle_waiting += [traci.vehicle.getAccumulatedWaitingTime(vehicle_id) for vehicle_id in vehicle_ids]

        # Identify the corresponding edge for this lane
        corresponding_edge = traci.lane.getEdgeID(lane)

        # Avoid duplicate data collection for the same edge
        if corresponding_edge not in edges:
            edges.append(corresponding_edge)
            throughputs.append(traci.edge.getLastStepVehicleNumber(corresponding_edge))  # Number of vehicles passing
            queue_lengths.append(traci.edge.getLastStepHaltingNumber(corresponding_edge))  # Number of halted vehicles
            Occupancies.append(traci.edge.getLastStepOccupancy(corresponding_edge))  # Lane occupancy percentage

    # Convert lists to NumPy arrays for efficient computation
    vehicle_speeds = np.array(vehicle_speeds)
    avg_speed = np.mean(vehicle_speeds) if vehicle_speeds.size > 0 else 0
    var_speed = np.var(vehicle_speeds) if vehicle_speeds.size > 0 else 0
    
    vehicle_waiting = np.array(vehicle_waiting)
    avg_waiting_time = np.mean(vehicle_waiting) if vehicle_waiting.size > 0 else 0
    var_waiting_time = np.var(vehicle_waiting) if vehicle_waiting.size > 0 else 0
        
    avg_throughput = np.mean(throughputs) if throughputs else 0
    avg_queue_length = np.mean(queue_lengths) if queue_lengths else 0
    avg_Occupancy = np.mean(Occupancies) if Occupancies else 0

    #clean memory
    del(vehicle_waiting)
    del(vehicle_speeds)
    del(edges)
    del(throughputs)
    del(queue_lengths)
    del(Occupancies)
    
    return (avg_speed, var_speed, avg_waiting_time, var_waiting_time, avg_throughput, avg_queue_length, avg_Occupancy)



def reward_proposed(agent,action, single_state, last_action):
    """
    Calculates the independent reward for a single traffic light based on various traffic metrics.
    
    Parameters:
    agent (String): The agent id.
    single_state (tuple): Contains traffic state metrics in the following order:
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
    w1 = 0.6
    w2 = 0.4
    scale_speed = 0.23333333333333334
    scale_waiting = 0.6666666666666666
    scale_efficiency = 0.1
    eta = 1e-6  # To prevent division by zero

    avg_speed, var_speed, avg_waiting_time, var_waiting_time, avg_throughput, avg_queue_length, avg_Occupancy = getStateProposed(agent)

    # Speed reward
    speed_term = np.log(1 + avg_speed / (var_speed + eta))

    # Waiting time penalty
    waiting_term = 1 / (1 + avg_waiting_time * var_waiting_time)

    # Traffic efficiency metric
    traffic_efficiency = w1 * avg_throughput - w2 * avg_queue_length 

    # Final reward formula
    independent_part_reward = scale_speed * speed_term - (scale_waiting * waiting_term) + (scale_efficiency * traffic_efficiency)

    return independent_part_reward



def get_detailed_road_literature( agent):
        """
        Get waiting times of vehicles for each controlled lane of a traffic light.

        Returns:
            list: List of accumulated waiting times.
        """
        waitings_s = []
        lanes = traci.trafficlight.getControlledLanes(agent)
        for lane in lanes:
            ids = traci.lane.getLastStepVehicleIDs(lane)
            for id in ids:
                waitings_s.append(traci.vehicle.getAccumulatedWaitingTime(id))
        return waitings_s


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
    
    road_inf = get_detailed_road_literature(agent)
    if not road_inf:  # This covers None or an empty list
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