import traci
import numpy as np
from Observations.obs import obs


class SumoObservation(obs):
    """
    Base SUMO observation class.
    Inherits from `obs` and serves as the foundation for traffic environment observation.
    """
    pass


class DefaultObservation(SumoObservation):
    """
    Default observation handler for SUMO environments.
    Collects metrics such as speed, waiting time, queue length, throughput, and occupancy.
    """

    def __init__(self, env, agent_id, lanes, edges):
        """
        Initialize the observation object for a single agent.

        Args:
            env: Environment instance containing SUMO connection and config.
            agent_id (str): Identifier of the agent (traffic light ID).
            lanes (list[str]): Controlled lane IDs for this agent.
            edges (list[str]): Controlled edge IDs for this agent.
        """
        super().__init__()
        self.env = env
        self.id = agent_id
        self.lanes = lanes
        self.edges = edges
        self.vehicles = {}  # Tracks per-vehicle lane waiting times.

    # -------------------
    # Internal Helpers
    # -------------------
    def _get_last_step_vehicle_ids(self, lane):
        """Return a list of vehicle IDs currently on the given lane."""
        return traci.lane.getLastStepVehicleIDs(lane)

    def _get_current_lane_id(self, veh_id):
        """Return the current lane ID of a given vehicle."""
        return traci.vehicle.getLaneID(veh_id)

    def _get_accumulated_waiting_time(self, veh_id):
        """Return total accumulated waiting time for the given vehicle."""
        return traci.vehicle.getAccumulatedWaitingTime(veh_id)

    # -------------------
    # Waiting Time Metrics
    # -------------------
    def get_waiting_time(self):
        """
        Compute waiting time per controlled lane.

        Tracks vehicle lane changes to avoid double-counting waiting times.
        """
        wait_time_per_lane = []

        for lane in self.lanes:
            veh_list = self._get_last_step_vehicle_ids(lane)
            lane_wait_time = 0.0

            for veh in veh_list:
                veh_lane = self._get_current_lane_id(veh)
                total_wait = self._get_accumulated_waiting_time(veh)

                if veh not in self.vehicles:
                    self.vehicles[veh] = {veh_lane: total_wait}
                else:
                    # Subtract time already counted in other lanes
                    prev_wait = sum(
                        self.vehicles[veh][ln]
                        for ln in self.vehicles[veh]
                        if ln != veh_lane
                    )
                    self.vehicles[veh][veh_lane] = total_wait - prev_wait

                lane_wait_time += self.vehicles[veh][veh_lane]

            wait_time_per_lane.append(lane_wait_time)

        return np.array(wait_time_per_lane)

    def get_avg_waiting_time(self):
        """Return average waiting time across controlled lanes."""
        wt = self.get_waiting_time()
        return wt.mean() if wt.size > 0 else 0.0

    def get_var_waiting_time(self):
        """Return variance of waiting times across controlled lanes."""
        wt = self.get_waiting_time()
        return wt.var() if wt.size > 0 else 0.0

    # -------------------
    # Speed Metrics
    # -------------------
    def get_vehicle_list(self):
        """Return all vehicles currently in the controlled lanes."""
        vehs = []
        for lane in self.lanes:
            vehs.extend(self._get_last_step_vehicle_ids(lane))
        return vehs

    def get_avg_speed(self):
        """Return average speed of vehicles in controlled lanes."""
        vehs = self.get_vehicle_list()
        if not vehs:
            return 0.0  # Default when no vehicles present
        return np.mean([traci.vehicle.getSpeed(v) for v in vehs])

    def get_var_speed(self):
        """Return variance of vehicle speeds in controlled lanes."""
        vehs = self.get_vehicle_list()
        if not vehs:
            return 0.0
        return np.var([traci.vehicle.getSpeed(v) for v in vehs])

    # -------------------
    # Queue Length
    # -------------------
    def get_average_queue_length(self):
        """Return the average queue length across controlled lanes."""
        return np.mean([traci.lane.getLastStepHaltingNumber(l) for l in self.lanes])

    # -------------------
    # Throughput
    # -------------------
    def get_average_throughput(self):
        """Return average number of vehicles passing per step (per edge)."""
        throughputs = [traci.edge.getLastStepVehicleNumber(e) for e in self.edges]
        return np.mean(throughputs) if throughputs else 0.0

    # -------------------
    # Occupancy
    # -------------------
    def get_average_occupancy(self):
        """Return average occupancy percentage of controlled edges."""
        occs = [traci.edge.getLastStepOccupancy(e) for e in self.edges]
        return np.mean(occs) if occs else 0.0

    # -------------------
    # State Representation
    # -------------------
    def len_sensors(self):
        """Return number of observation features."""
        return 7

    def get_state_space(self):
        """
        Return a tuple of traffic state metrics:
            (avg_speed, var_speed,
             avg_waiting_time, var_waiting_time,
             avg_throughput, avg_queue_length,
             avg_occupancy)
        """
        return (
            self.get_avg_speed(),
            self.get_var_speed(),
            self.get_avg_waiting_time(),
            self.get_var_waiting_time(),
            self.get_average_throughput(),
            self.get_average_queue_length(),
            self.get_average_occupancy()
        )

  
 