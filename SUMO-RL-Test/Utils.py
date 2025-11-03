import xml.etree.ElementTree as ET
import numpy as np
import os
from Observations.sumo_obs import DefaultObservation # wrap the state space defined here instead rewrite it from scratch
import traci
from sumo_rl.environment.traffic_signal import TrafficSignal
from sumo_rl.environment.observations import ObservationFunction
from rewards import reward_proposed,reward_liter
from gymnasium.spaces import Box

def sumo_rl_literature_reward(ts):
    lambda_ = 0.15
    Tw = 20
    alpha = 2

    waiting_times = ts.get_accumulated_waiting_time_per_lane()

    terms = []
    for wi in waiting_times:
        if wi is not None:
            val = lambda_ * (1 - ((wi / Tw) ** alpha))
            terms.append(val)

    if not terms:
        return 0.0

    return float(np.mean(terms))

def sumo_rl_proposed_reward(ts):
        lanes = traci.trafficlight.getControlledLanes(ts.id)
        edges = [traci.lane.getEdgeID(l) for l in lanes]
        obs =  DefaultObservation(None,ts.id,lanes,edges)
        return reward_proposed(ts.id,obs,None,None)
    

class SumoRL_State_Wrapper(ObservationFunction):
    """Abstract base class for observation functions."""

    def __init__(self, ts: TrafficSignal):
        """Initialize observation function."""
        self.ts = ts
        self.lanes = traci.trafficlight.getControlledLanes(self.ts.id)
        self.edges = [traci.lane.getEdgeID(l) for l in self.lanes]
        self.obs =  DefaultObservation(None,self.ts.id,self.lanes,self.edges)

    def __call__(self):
        return np.array(self.obs.get_state_space(),dtype=np.float32)

    def observation_space(self):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),
            dtype=np.float32
        )

def get_sumo_statics(data_path=""):
        """
        Parse SUMO-generated statistics file.

        Args:
            data_path (str): Path to SUMO output directory.

        Returns:
            dict | None: Parsed statistics or None if failed.
        """
        file_path = os.path.join(data_path, "osm.statistics.xml")
        try:
            root = ET.parse(file_path).getroot()
            vehicles = root.find("vehicles").attrib
            trips = root.find("vehicleTripStatistics").attrib
            return {
                "waiting_time": float(trips["waitingTime"]),
                "speed": float(trips["speed"]),
                "waiting_vehicles": int(vehicles["waiting"]),
                "time_loss": float(trips["timeLoss"]),
                "depart_delay": float(trips["departDelay"])
            }
        except (ET.ParseError, KeyError, AttributeError, ValueError):
            print(f"Failed to parse {file_path}")
            return None