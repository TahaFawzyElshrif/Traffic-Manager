from Connections.Connection import Connection
from Sensors import getSumoSensors_full ,len_sensors ,len_optimized_sensors
import traci
import os
import json
from dotenv import load_dotenv
import uuid
import random
import numpy as np
import subprocess
import pandas as pd

# Load secrets from the environment file
load_dotenv("keys.env")

# Retrieve SUMO_HOME path from environment variables
sumo_home = str(os.getenv("sumo_home"))
 
class SumoConnection(Connection):
    """
    Handles the connection to the SUMO traffic simulator.

    This class manages initializing, resetting, and closing the connection to SUMO.

    Attributes:
        cmd (list): The command used to start SUMO.
        traci_conn (traci.Connection): The active connection to SUMO.
        gui (bool): Indicates whether SUMO is running in GUI mode.
    """

    def __init__(self, cmd):
        """
        Initializes a new SUMO connection.

        Parameters:
            cmd (list): The command used to start SUMO.
        """
        super().__init__()
        self.cmd = cmd
        self.traci_conn = None
        self.gui = False
        self.initialize()

    def initialize(self):
        """
        Starts or reuses a SUMO connection.

        If SUMO_HOME is set, it assigns it to the environment variable.
        It checks if there is an existing connection; if found, it uses it;
        otherwise, it starts a new connection.
        """
        if sumo_home is not None:
            os.environ["SUMO_HOME"] = sumo_home

        # Determine if SUMO should run in GUI mode
        if "-gui" in self.cmd[0]:
            self.gui = True
        else:
            self.gui = False

        try:
            # Try to get an existing connection
            self.traci_conn = traci.getConnection()
            if self.traci_conn is not None:
                print("Found existing connection to SUMO and used it. To make a new connection, reset ✔")
            else:
                print("No connection found, creating a new connection ❌")
                traci.start(self.cmd)
                self.traci_conn = traci.getConnection()
        except traci.exceptions.TraCIException:
            print("No connection found, creating a new connection ❌")
            traci.start(self.cmd)
            self.traci_conn = traci.getConnection()

    def close(self):
        """
        Closes the SUMO connection if one exists.
        """
        try:
            if traci.getConnection() is not None:
                traci.close()
        except traci.exceptions.TraCIException:
            print("No active connection to close.")

    def reset(self):
        """
        Resets the SUMO connection by closing it and reinitializing.
        """
        self.close()
        self.initialize()

    def done_cond(self, max_sumo_step):
        """
        Check if the simulation has reached or passed the maximum step.

        Parameters:
            max_sumo_step (float): The time limit of the simulation.

        Returns:
            bool: True if simulation time has reached the maximum step.
        """
        return (max_sumo_step <= traci.simulation.getTime())

    def do_steps_duration(self, duration, max_sumo_step, agent):
        """
        Perform a sequence of simulation steps for a specified duration.

        Returns:
            bool: True if simulation is done before completing duration.
        """
        for i in range(duration):
            if self.done_cond(max_sumo_step):
                return True
            # Add traffic every 10 seconds
            #if (traci.simulation.getTime()) % 10 == 0:
            #    self.add_traffic(agent, traffic_scale)

            traci.simulationStep()
        return False

    def do_step_one_agent(self, agent, new_action, duration, max_sumo_step,step_index):
        """
        Executes an action for a single agent and steps the simulation.

        Returns:
            bool: True if simulation is done after the steps.
        """
        traci.trafficlight.setRedYellowGreenState(agent, new_action)
        result_duration_cond = self.do_steps_duration(duration, max_sumo_step, agent)
        self.update_metrics_each_step(agent,step_index)
        if result_duration_cond:
            return True

    def get_detailed_road_literature(self, agent):
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
    
    def get_improved_road_proj(self, agent):
        """
        Get sensor data for a traffic light without depending on agent_data.

        Returns:
            np.array: Aggregated state observation including occupancy,
                      vehicle counts, halting numbers, teleports, and emergency stops.
        """
         
        if not hasattr(self, '_vehicle_speeds_history'):
            self._vehicle_speeds_history = {}
        #print(f"self._vehicle_speeds_history is {self._vehicle_speeds_history}")
        lanes = traci.trafficlight.getControlledLanes(agent)
        edges = []
        for lane in lanes:
            edge = traci.lane.getEdgeID(lane)
            if edge not in edges:
                edges.append(edge)

        vehicles_started_to_teleport = traci.simulation.getStartingTeleportNumber()
        emergency_stops = 0

        # get cuurent vehicles speed
        current_vehicles = {}
        for lane in lanes:
            ids = traci.lane.getLastStepVehicleIDs(lane)
            for id in ids:
                current_vehicles[id] = traci.vehicle.getSpeed(id)
        
        # compare with previous speeds to get emergency stops
        for vehicle_id in current_vehicles:
            if vehicle_id in self._vehicle_speeds_history:
                speed_diff = self._vehicle_speeds_history[vehicle_id] - current_vehicles[vehicle_id]
                if speed_diff > 4.5:  # عتبة التوقف الطارئ
                    emergency_stops += 1
        
        # update _vehicle_speeds_history
        self._vehicle_speeds_history = current_vehicles
        
        # make observation
        observation = []
        for e_id in edges:
            edge_values = [
                traci.edge.getLastStepOccupancy(e_id),
                traci.edge.getLastStepVehicleNumber(e_id),
                traci.edge.getLastStepHaltingNumber(e_id)
            ]
            observation.append(edge_values)

        if observation:
            observation = np.matrix(observation).mean(0).tolist()[0]
        else:
            observation = [0, 0, 0]

        observation.append(vehicles_started_to_teleport)
        observation.append(emergency_stops)

        return np.array(observation)
    
    def getCurrentState(self, agent):
        """
        Get the current sensor state for a traffic light agent.

        Returns:
            list: The observation from SUMO sensors.
        """
        return getSumoSensors_full(agent)

    def getLenSensors(self):
        """
        Return the number of sensors (length of the observation vector).

        Returns:
            int: Length of the observation vector.
        """
        return len_sensors

    def update_metrics_each_step(self,tl_id,step_number):
        """
        Update metrics during each simulation step.

        Tracks delays, waiting time, speed, and time loss.
        """
        if step_number==0:
            self.reset_metrices()


        # Get all lanes controlled by the traffic light
        lanes = traci.trafficlight.getControlledLanes(tl_id)
        lane_vehicles = set()
        for lane_id in lanes:
            lane_vehicles.update(traci.lane.getLastStepVehicleIDs(lane_id))

        departed_vehicles = traci.simulation.getDepartedIDList()
        for veh_id in departed_vehicles:
            if veh_id not in lane_vehicles:
                continue
            try:
                intended_depart = float(traci.vehicle.getParameter(veh_id, "departTime"))
                actual_depart = traci.simulation.getTime()
                delay = actual_depart - intended_depart
                self.total_depart_delay += delay
                self.departed_count += 1
            except:
                pass

        active_vehicles = traci.vehicle.getIDList()
        step_waiting_time = 0
        step_speed = 0
        step_time_loss = 0
        step_vehicle_count = 0

        for veh_id in active_vehicles:
            if veh_id not in lane_vehicles:
                continue
            waiting_time = traci.vehicle.getWaitingTime(veh_id)
            step_waiting_time += waiting_time

            speed = traci.vehicle.getSpeed(veh_id)
            step_speed += speed

            try:
                time_loss = traci.vehicle.getTimeLoss(veh_id)
                step_time_loss += time_loss
            except:
                pass

            step_vehicle_count += 1

        self.total_waiting_time += step_waiting_time
        self.total_speed += step_speed
        self.total_time_loss += step_time_loss
        self.vehicle_count += step_vehicle_count
    
    def getTime(self):
        return traci.simulation.getTime()
    def get_sumo_statics(self,python_path, data_path):
        # Construct the paths using os.path.join (automatically handles path separator)
        script_path = os.path.join(python_path, 'tools', 'xml', 'xml2csv.py')
        data_file_path = os.path.join(data_path, "osm.statistics.xml")

        script_path = os.path.normpath(script_path)
        data_file_path = os.path.normpath(data_file_path)

        # Determine the correct Python executable based on the OS
        python_executable = 'python' if os.name == 'nt' else 'python3'  # 'nt' is for Windows

        # Check if the script exists before running
        if not os.path.exists(script_path):
            print(f"Error: Script not found at {script_path}")
            return
        
        # Prepare the command to run the script
        command = [python_executable, script_path, data_file_path]

        # Run the script using subprocess
        try:
            subprocess.run(command, check=True)
            print(f"Successfully ran {script_path} with {data_file_path}")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred: {e}")
        except FileNotFoundError:
            print(f"File not found: {script_path}")

        data = pd.read_csv(os.path.join(data_path, "osm.statistics.csv"), delimiter=';')

        Values_dict = dict({
            'Waiting_Time':float(data['vehicleTripStatistics_waitingTime'].dropna().iloc[0]) ,
            'Speed':float(data['vehicleTripStatistics_speed'].dropna().iloc[0])   ,
            'Waiting_Vehicles':float(data['vehicles_waiting'].dropna().iloc[0])   ,
            'TimeLoss':float(data['vehicleTripStatistics_timeLoss'].dropna().iloc[0])   ,
            'Depart_Delay':float(data['vehicleTripStatistics_departDelay'].dropna().iloc[0])   ,  

        })
        return Values_dict

    def collect_final_metrics(self):
        """
        Calculate final metrics after simulation is complete.

        Returns:
            dict: Dictionary with average metrics like waiting time, speed, etc.
        """
        metrics = {
            "waiting_time": 0,
            "speed": 0,
            "depart_delay": 0,
            "time_loss": 0,
            "waiting_vehicles": 0
        }

        if self.vehicle_count > 0:
            metrics["waiting_time"] = self.total_waiting_time / self.vehicle_count
            metrics["speed"] = self.total_speed / self.vehicle_count
            metrics["depart_delay"] = self.total_depart_delay / self.departed_count if self.departed_count > 0 else 0
            metrics["time_loss"] = self.total_time_loss / self.vehicle_count
            
        return metrics
    
    

    
    def add_traffic(self,tl_id, n):
        """
        Adds random vehicles to a specified traffic light-controlled area.

        This function selects a random lane controlled by the given traffic light ID and spawns vehicles
        on that lane while ensuring compatibility with lane restrictions.

        Parameters:
            tl_id (str): The traffic light ID whose controlled lanes are used.
            n (int): The number of vehicles to add.
        """
        controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)

        if not controlled_lanes:
            return    

        for i in range(n):
            lane = random.choice(controlled_lanes)
            edge_of_lane = traci.lane.getEdgeID(lane)
            
            # Randomly select a vehicle type
            type_v = random.choice(traci.vehicletype.getIDList())

            # Ensure the selected vehicle type is allowed in the chosen lane
            if traci.vehicletype.getVehicleClass(type_v) in traci.lane.getAllowed(lane):
                route_of_edge = "route_" + str(edge_of_lane)

                # If the route does not exist, create it
                if route_of_edge not in traci.route.getIDList():
                    traci.route.add(route_of_edge, [edge_of_lane])
                
                # Generate a unique vehicle ID
                vehicle_id = str(uuid.uuid4())

                # Randomly position the vehicle within the lane to prevent congestion
                pos = random.uniform(0, traci.lane.getLength(lane) - 5)

                # Add the vehicle to the simulation and move it to the chosen position
                traci.vehicle.add(vehicle_id, route_of_edge, typeID=type_v)
                traci.vehicle.moveTo(vehicle_id, lane, pos)  # Move to lane start

    def set_traffic_scale(self ,scale):
        """
        Set the Scale of traffic for simulation steps
        """
        traci.simulation.setScale(scale)
