from sumoEnv import SumoEnv
import sys
import argparse
from theParser import sumoParser


cmd = argparse.ArgumentParser()
arguments = sumoParser(cmd)

max_num_steps=arguments.i
scenario=arguments.s

class config():
    def __init__(self, sumoBinary, sumoCmd, sumo_home=None):
        self.sumoBinary = sumoBinary
        self.sumoCmd = sumoCmd
        self.sumo_home = sumo_home

if scenario == "lust":
    #path = "D:/Intelligent-Traffic-Analysis/LuSTScenario-master/LuSTScenario-master/scenario/due.static.sumocfg"
    path="D:/study/Projects/faculty/graduation project/old_project/Lust/scenario/due.static.sumocfg"
    #path="Lust/due.static.sumocfg"
else:
    #path = "D:/Traffic-Management-System-CCE/Alex/osm.sumocfg"
    path="D:/study/Projects/faculty/graduation project/old_project/AIST/osm.sumocfg"
    #path="Alex/osm.sumocfg"

SumoGUIConfig = config(
    "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo-gui.exe",
    ["C:/Program Files (x86)/Eclipse/Sumo/bin/sumo-gui.exe", "-c", path, "--duration-log.statistics","--mesosim"],
    "C:/Program Files (x86)/Eclipse/Sumo"
)

env = SumoEnv(SumoGUIConfig, [], lambda x: x)



for i in range(max_num_steps):
    env.simulationStepOnly()

env.close()



