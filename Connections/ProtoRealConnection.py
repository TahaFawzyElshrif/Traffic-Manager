from Sensors import getSumoSensors_full ,len_sensors ,len_optimized_sensors
from dotenv import load_dotenv
import numpy as np
from Connections.Connection import Connection
from Connections.Proto_Connection_Utils.ardinuo import *
from Connections.Proto_Connection_Utils.Video import *
import math

class ProtoRealConnection(Connection):
    """
    Real connection used for testing purposes.
    """

    def __init__(self,port,video_path,yolo_path,yellow_seconds=15,one_frame_processing=.56,DISPLAY_WIDTH = 800,DISPLAY_HEIGHT = 450,video_output_path = '',window_name = 'STREET STATE CAMERA',thresould_speed = .01):
        """
        Initializes metrics for traffic simulation.
        one_frame_processing =.56 for yolo11 , .2 for pruned yolo11 version
        thresould_speed in m/s
        """
        self.port = port
        self.DISPLAY_WIDTH = DISPLAY_WIDTH
        self.DISPLAY_HEIGHT = DISPLAY_HEIGHT
        self.video_output_path = video_output_path
        self.window_name = window_name
        self.thresould_speed = thresould_speed
        self.video_path=video_path
        self.yolo_path = yolo_path
        self.ardinuo = None
        self.one_frame_processing = one_frame_processing #.56 for yolo11 , .2 for pruned yolo11 version
        self.lane_id = 0
        self.yellow_seconds =  yellow_seconds
        self.ardinuo_second_error = 1
        self.initialize()

        # intiail video step 
        self._ret, self._frame_vid = self.lane_video_estimatior.cap.read()   
        self._estimate = self.lane_video_estimatior.speedestimator(self._frame_vid)
        self._last_real_action = 'g'

    def initialize(self):
        self.ardinuo = ardinuo(self.lane_id,self.port)

        self.lane_video_estimatior = Video_Estimatior(self.lane_id,self.video_path,self.yolo_path,self.thresould_speed,self.window_name,self.video_output_path,self.one_frame_processing,self.DISPLAY_WIDTH ,self.DISPLAY_HEIGHT )


    def getTime(self):
        return 0

    
    def close(self):
        """
        Closes the connection.
        """
        self.lane_video_estimatior.close_vid()
        self.ardinuo.close_arduino()

    def reset(self):
        """
        Resets the simulation.
        """
        self.close()
        self.initialize()

    def done_cond(self):
        """
        Checks if the simulation is done.
        """
        return False # Never done
    
    def getLenSensors(self):
        """
        Retrieves the number of sensors.
        """
        return 7
    
    def update_video(self,action,duration,esimation):
        if (action == 'g'):
                for i in range(duration *self.lane_video_estimatior.fps):
                    # ⚠️ Just be sure estimation happens often enough to correct overcounting.
                    # ⚠️ This is a bit of a hack, but it works for now.
                    self.ret_, self._frame_vid = self.lane_video_estimatior.cap.read()   
                    if not self.ret_:
                        print("end")
                    # break
                    self.lane_video_estimatior.render(self._frame_vid)

                    if esimation:
                        estimate = self.lane_video_estimatior.speedestimator(self._frame_vid)
                        #frame_est = estimate.plot_im         
                    
                    self.lane_video_estimatior.update_waiting_video()

        else:
                self.lane_video_estimatior.update_waiting_red_video(duration) 
                self.lane_video_estimatior.render(self._frame_vid)
                self.lane_video_estimatior.render_frame_for_time(duration)
                
        


    def do_signal(self,action_c,duration_wanted):
        duration_wanted_without_ardinuo = duration_wanted - self.ardinuo_second_error
        estimated_max_time_to_estimate_from_video = int(duration_wanted_without_ardinuo / int(self.lane_video_estimatior.one_second_processing) ) 

        start =time.time()
        self.update_video(action_c,estimated_max_time_to_estimate_from_video,True) 
        end =time.time()
        remain_of_duration = math.floor(duration_wanted - (end-start))
        self.update_video(action_c,remain_of_duration,False) 
        
        
    
    def do_steps_duration(self, action_lane_i,duration, max_sumo_step, agent, traffic_scale):
        """
        Performs simulation steps for a given duration.
        """
        
        self.ardinuo.send_command(action_lane_i,duration)
        self.do_signal(action_lane_i,duration)
        

    

    def do_step_one_agent(self, agent, new_action, duration, max_sumo_step, traffic_scale):
        """
        Performs a simulation step for a single agent.
        """
        action_lane_i = new_action[0]

        if self._last_real_action == action_lane_i:

                self.do_steps_duration( action_lane_i , duration, max_sumo_step, agent, traffic_scale)
                

        else:
                self.do_steps_duration( 'y' , self.yellow_seconds, max_sumo_step, agent, traffic_scale)
                
                self.do_steps_duration( action_lane_i , duration, max_sumo_step, agent, traffic_scale)

        self._last_real_action = action_lane_i



    def getCurrentState(self, agent):
        """
        Retrieves the current state for a given agent.
        """
        return self.lane_video_estimatior.get_State_video()



    def get_detailed_road_literature(self, agent):
        """
        Retrieves detailed road literature data for an agent.
        """
        return self.lane_video_estimatior.get_waiting_video()


