from __future__ import division
import sys
import os
import itertools
import os
import numpy as np
import csv
import math
import random
try:
    import traci
except ImportError:
    if "SUMO_HOME" in os.environ:
        sys.path.append(
            os.path.join(os.environ["SUMO_HOME"], "tools")
        )
        import traci
    else:
        raise EnvironmentError("Importing Traci failed or SUMO was not found in the system")

class Agent_Info(object):
    pass


class SumoEnv():

    def actionCount(self, tl_id):
        agent = self.agent_data[tl_id]
        return len(list(self.action_spaces[-1]))  # agent.tl_count]))

    def stateCount(self, tl_id):  # i think not important or need constant ok????!! may be to same way as actioncount
        return 5

    def __init__(self, config, traffic_light_info, reward_function, type_Action, seed=0):
        # Set random seed if provided
        self.seed_value = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.action_spaces = [None]  # * 67  
        self.edges = []
        self.agent_data = {}
        self.lanes = []
        self.reward_function = reward_function
        self.config = config
        self.startTraci()
        self.current_step = 0

        for (count, tl_id) in (traffic_light_info):  # understand class for info of each controlled traffic light
            info = Agent_Info()
            info.tl_count = count
            info.tl_id = tl_id
            info.vehicles_last_step = {}
            info.last_action = None
            info.lanes = traci.trafficlight.getControlledLanes(tl_id)
            info.edges = self.getSumoEdgeInformationFromTraci(tl_id)
            # also with agent_info.observation, agent_info.reward as in actionResults (may be added later)
            
            self.type_Action = type_Action
            self.intializeActionSpace(count)  # selff
            self.agent_data[tl_id] = info
            self.last_agent_id = tl_id  # as only single agent so to kep track of this agent

    def getSumoEdgeInformationFromTraci(self, tl_id):
        lanes = traci.trafficlight.getControlledLanes(tl_id)
        result = []
        for lane in lanes:
            result.append(traci.lane.getEdgeID(lane))
        return result
    
    # actionspace is rgyGu one char of it for each traffic light, state is info of traffic defined in getSensors()
    def intializeActionSpace(self, size):
        if self.action_spaces[-1] is not None:  # action_spaces[size]
            return
        if self.type_Action == "High":
            space = ["r"*size, "g"*size] if (size > 0) else ["r", "g"]
        
        elif self.type_Action == "Grouped":
            space = list(map("".join, itertools.product("rg", repeat=4)))
        
        else:  # self.type_Action=="Full":
            space = list(map(''.join, itertools.product("rg", repeat=size)))  # all this, as it's action space, not one action, space for all controlled traffic

        self.action_spaces[-1] = list(space)  # -----> what in self.action_spaces : is dic : for each size :the corresponding space
        # ---> so even the function is in loop it is not Keep Only the Last Iteration, and so the first check here, may to easy computation

    def actionResults(self, tl_id):
        agent_info = self.agent_data[tl_id]
        return agent_info.observation, agent_info.reward, False, {}

    def simulationStepOnly(self):  # basic sumo
        traci.simulationStep()

    def simulationStepNoObservations(self):
        self.performActions()
        traci.simulationStep()

    def step(self):
        self.performActions()
        traci.simulationStep()
        self.current_step += 1
        self.makeObservations()
        self.computeRewards()
        self.storeLastActions()

    def storeLastActions(self):
        for tl_id, agent in self.agent_data.items():
            action_space = self.action_spaces[-1]  # agent.tl_count]#!!!!  # ----->ex use of action space 
            agent.last_action = action_space[agent.action]
        
    def get_lane_direction(self, lane_id):
        """Determine the primary cardinal direction of a lane in SUMO."""
        x_start, y_start = traci.lane.getShape(lane_id)[0]  # First coordinate
        x_end, y_end = traci.lane.getShape(lane_id)[-1]     # Last coordinate

        angle = math.degrees(math.atan2(y_end - y_start, x_end - x_start))

        # Optimized angle-based direction mapping
        return "E" if -45 <= angle < 45 else "N" if 45 <= angle < 135 else "W" if angle >= 135 or angle < -135 else "S"

    def get_all_lanes_action(self, action):
        """Map SUMO controlled lanes to the corresponding action signals."""
        direction_map = {"N": 0, "E": 1, "S": 2, "W": 3}
        
        controlled_lanes = traci.trafficlight.getControlledLanes(self.last_agent_id)
        real_action_dict = {}

        try:
            real_action_list = [
                action[direction_map[dir]] for lane in controlled_lanes if (dir := self.get_lane_direction(lane)) in direction_map
            ]
        except IndexError:
            raise ValueError(f"Invalid action index for action: {action}")

        return "".join(real_action_list), real_action_dict
    
    def get_real_action(self, action_agent_lanes):
        real_action = self.get_all_lanes_action(action_agent_lanes)[0]
        return real_action
    
    def computeRewards(self):
        for tl_id, agent in self.agent_data.items():
            if self.type_Action == "Grouped":
                action = self.get_real_action(self.action_spaces[-1][agent.action])  # agent.tl_count
            else:
                action = self.action_spaces[-1][agent.action]
            agent.reward = self.reward_function(tl_id, action, agent.observation, agent.last_action)

    def makeObservations(self):
        for tl_id, agent in self.agent_data.items():
            agent.observation = self.get_improved_road_proj(agent.tl_id)  # self.getSensors(agent.tl_id)
            
           
    def performActions(self):
        for tl_id, agent in self.agent_data.items():
            action_space = self.action_spaces[-1]  # agent.tl_count]
            action_space = list(action_space)

            if self.type_Action == "Grouped":
                action = self.get_real_action(action_space[agent.action])
            else:
                action = action_space[agent.action]

            traci.trafficlight.setRedYellowGreenState(tl_id, action)

    def setAction(self, action, tl_id):
        self.agent_data[tl_id].action = action

    def get_improved_road_proj(self, agent):
        # حفظ سرعات المركبات بين الاستدعاءات
        if not hasattr(self, '_vehicle_speeds_history'):
            self._vehicle_speeds_history = {}
        
        # print(f"_vehicle_speeds_history: {self._vehicle_speeds_history}")
        
        lanes = traci.trafficlight.getControlledLanes(agent)
        edges = []
        for lane in lanes:
            edge = traci.lane.getEdgeID(lane)
            if edge not in edges:
                edges.append(edge)

        vehicles_started_to_teleport = traci.simulation.getStartingTeleportNumber()
        emergency_stops = 0

        # الحصول على سرعات المركبات الحالية
        current_vehicles = {}
        for lane in lanes:
            ids = traci.lane.getLastStepVehicleIDs(lane)
            for id in ids:
                current_vehicles[id] = traci.vehicle.getSpeed(id)
        
        # مقارنة السرعات مع السابقة لاكتشاف التوقفات الطارئة
        for vehicle_id in current_vehicles:
            if vehicle_id in self._vehicle_speeds_history:
                speed_diff = self._vehicle_speeds_history[vehicle_id] - current_vehicles[vehicle_id]
                if speed_diff > 4.5:  # عتبة التوقف الطارئ
                    emergency_stops += 1
        
        # تحديث سجل السرعات
        self._vehicle_speeds_history = current_vehicles
        
        # مكون الملاحظة  
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
    
    def getSensors(self, tl_id):
        edges = self.agent_data[tl_id].edges
        vehicles_started_to_teleport = traci.simulation.getStartingTeleportNumber()
        lanes = self.agent_data[tl_id].lanes
        vehicles_last_step = self.agent_data[tl_id].vehicles_last_step
        emergency_stops = 0
        vehicles = {}

        for lane in lanes:
            ids = traci.lane.getLastStepVehicleIDs(lane)
            for id in ids:
                speed = traci.vehicle.getSpeed(id)
                vehicles[id] = speed
                if id in vehicles_last_step:
                    if vehicles_last_step[id] - speed > 4.5:
                        emergency_stops += 1
        self.agent_data[tl_id].vehicles_last_step = vehicles

        observation = []
        for e_id in edges:
            edge_values = [traci.edge.getLastStepOccupancy(e_id), traci.edge.getLastStepVehicleNumber(e_id), traci.edge.getLastStepHaltingNumber(e_id)]
            observation.append(edge_values)
        
        if observation:
            observation = np.matrix(observation).mean(0).tolist()[0]
        else:
            observation = [0, 0, 0]
        
        observation.append(vehicles_started_to_teleport)
        observation.append(emergency_stops)

        return np.array(observation)

    def close(self):
        traci.close()

    def startTraci(self):
        if self.config.sumo_home is not None:
            os.environ["SUMO_HOME"] = self.config.sumo_home
        if "-gui" in self.config.sumoCmd[0]:
            self.gui = True
        else:
            self.gui = False
        traci.start(self.config.sumoCmd)

    def emptyState(self, tl_id):
        return np.zeros(self.stateCount(tl_id))