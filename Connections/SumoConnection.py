import os
import json
import uuid
import random
import numpy as np
import pandas as pd
import traci
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
import sumolib
from Connections.Connection import Connection
from Sensors import getSumoSensors_full, len_sensors, len_optimized_sensors


# Load secrets and environment variables
load_dotenv("keys.env")
sumo_home = str(os.getenv("sumo_home"))


class SumoConnection(Connection):
    """
    Manages a connection to the SUMO traffic simulator using TraCI.
    
    Provides utility methods for retrieving lane/edge information, 
    executing actions for traffic light agents, collecting metrics,
    and controlling simulation flow.
    """


    def __init__(self, path_cfg, step_size=1, log_file=None, begin_time=0, end_time=None,
                time_to_teleport=-1, waiting_time_memory=1000, max_depart=-1, seed=None):
        """
        Initialize the SUMO connection.

        Args:
            path_cfg (str): Path to the SUMO configuration (.cfg) file.
            step_size (int, optional): Simulation step length in seconds. Defaults to 1.
            log_file (str or None, optional): Path to the log file. If None, logging is disabled.
            begin_time (int or float, optional): Simulation start time. Defaults to 0.
            end_time (int or float or None, optional): Simulation end time(Exactly End time). If None, no end time is set.
            time_to_teleport (int, optional): Max stuck time before teleportation (-1 disables). Defaults to -1.
            waiting_time_memory (int, optional): Time window for waiting time calculation. Defaults to 1000.
            max_depart (int, optional): Max allowed vehicle departure delay (-1 disables). Defaults to -1.
            seed (int or None, optional): Random seed for simulation. If None, no seed is set.
        """
        super().__init__()

        sumo_binary = sumolib.checkBinary('sumo')

        # Build command list dynamically based on passed arguments
        self.cmd = [
            sumo_binary,
            "-c", path_cfg,
            "--step-length", str(step_size),
            "--max-depart-delay", str(max_depart),
            "--waiting-time-memory", str(waiting_time_memory),
            "--time-to-teleport", str(time_to_teleport),
            "--begin", str(begin_time)
        ]

        if end_time is not None:
            self.cmd += ["--end", str(end_time)]

        if log_file:
            self.cmd += ["--log", log_file, "--verbose", "true"]

        if seed is not None:
            self.cmd += ["--seed", str(seed)]

        self.traci_conn = None
        self.gui = False
        self.done = False
        self.initialize()
        self.end_time = end_time

    def initialize(self):
        """
        Start or reuse a SUMO connection.
        """
        if sumo_home:
            os.environ["SUMO_HOME"] = sumo_home

        self.gui = "-gui" in self.cmd[0]

        try:
            self.traci_conn = traci.getConnection()
            if self.traci_conn:
                print("✔ Found existing connection to SUMO.")
            else:
                print("❌ No connection found. Creating a new one...")
                traci.start(self.cmd)
                self.traci_conn = traci.getConnection()
        except traci.exceptions.TraCIException:
            print("❌ No connection found. Creating a new one...")
            traci.start(self.cmd)
            self.traci_conn = traci.getConnection()
        self.done = False
    def close(self):
        """
        Close the SUMO connection if it exists.
        """
        try:
            if traci.getConnection():
                traci.close()
        except traci.exceptions.TraCIException:
            print("No active connection to close.")

    def reset(self):
        """
        Reset the SUMO connection by closing and reinitializing.
        """
        self.close()
        self.initialize()

    def done_cond(self, max_sumo_step):
        """
        Check if the simulation has reached the max step.

        Args:
            max_sumo_step (float): Maximum simulation step.

        Returns:
            bool: True if reached, else False.
        """
        return traci.simulation.getTime() >= max_sumo_step

    def step(self):
        """
        Advance the simulation by one step.
        """
        traci.simulationStep()
        if traci.simulation.getTime() >= self.end_time:
            self.done = True
        if traci.simulation.getMinExpectedNumber() <= 0:
            self.done = True

    def do_step_one_agent(self, agent, new_action):
        """
        Execute a traffic light action for one agent.

        Args:
            agent (str): Traffic light ID.
            new_action (str): Traffic light phase state string.
        """
        traci.trafficlight.setRedYellowGreenState(agent, new_action)



    def get_improved_road_proj(self, agent):
        """
        Collect aggregated road data for a traffic light.

        Returns:
            np.array: [occupancy, vehicle count, halting count, teleports, emergency stops]
        """
        if not hasattr(self, '_vehicle_speeds_history'):
            self._vehicle_speeds_history = {}

        lanes = traci.trafficlight.getControlledLanes(agent)
        edges = list({traci.lane.getEdgeID(l) for l in lanes})

        # Emergency stops & teleports
        vehicles_started_to_teleport = traci.simulation.getStartingTeleportNumber()
        emergency_stops = 0

        current_vehicles = {
            veh_id: traci.vehicle.getSpeed(veh_id)
            for lane in lanes
            for veh_id in traci.lane.getLastStepVehicleIDs(lane)
        }

        for veh_id, speed in current_vehicles.items():
            if veh_id in self._vehicle_speeds_history:
                if self._vehicle_speeds_history[veh_id] - speed > 4.5:
                    emergency_stops += 1

        self._vehicle_speeds_history = current_vehicles

        # Observation metrics
        observation = [
            [
                traci.edge.getLastStepOccupancy(e_id),
                traci.edge.getLastStepVehicleNumber(e_id),
                traci.edge.getLastStepHaltingNumber(e_id)
            ]
            for e_id in edges
        ]

        if observation:
            observation = np.mean(observation, axis=0).tolist()
        else:
            observation = [0, 0, 0]

        observation.extend([vehicles_started_to_teleport, emergency_stops])
        return np.array(observation)

    def getCurrentState(self, agent):
        """
        Get the current SUMO sensor state for a traffic light.

        Args:
            agent (str): Traffic light ID.

        Returns:
            list: Sensor observation.
        """
        return getSumoSensors_full(agent)

    def getLenSensors(self):
        """
        Get number of sensors in observation vector.

        Returns:
            int: Sensor count.
        """
        return len_sensors



    def getTime(self):
        """
        Get the current simulation time.

        Returns:
            float: Simulation time.
        """
        return traci.simulation.getTime()

    def get_sumo_statics(self, data_path):
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
            return None

    def set_traffic_scale(self, scale):
        """
        Adjust traffic scale in the simulation.

        Args:
            scale (float): Traffic scaling factor.
        """
        traci.simulation.setScale(scale)
