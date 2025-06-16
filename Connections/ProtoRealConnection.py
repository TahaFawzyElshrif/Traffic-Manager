from Sensors import getSumoSensors_full ,len_sensors ,len_optimized_sensors
from dotenv import load_dotenv
import numpy as np
from Connections.Connection import Connection
from Connections.Proto_Connection_Utils.ardinuo import *
from Connections.Proto_Connection_Utils.Video import *
import math
import requests
import requests
import time
import threading
import paho.mqtt.client as mqtt
import random
status_free = "FREE"
status_emr = "EMR"
status_acc = "ACC"
status_chk = "CHK"
MQTT_PARTOPIC_state="state"
MQTT_SUBTOPIC_reply="reply"
MQTT_SUBTOPIC_msg="msg"
route_pause= 'pause'
route_resume= 'resume'

class ProtoRealConnection(Connection):
    """
    Real connection used for testing purposes.
    """

    def __init__(self,ardinuo_port,mqtt_port,agent,global_url,broker,yolo_path,yellow_seconds=15,one_frame_processing=.56,thresould_speed = .01):
        """
        Initializes metrics for traffic simulation.
        one_frame_processing =.56 for yolo11 , .2 for pruned yolo11 version
        thresould_speed in m/s
        """
        self.ardinuo_port = ardinuo_port
        self.mqtt_port = mqtt_port

        self.thresould_speed = thresould_speed
        self.global_url=global_url
        self.yolo_path = yolo_path
        self.ardinuo = None
        self.one_frame_processing = one_frame_processing #.56 for yolo11 , .2 for pruned yolo11 version
        self.lane_id = 0
        self.yellow_seconds =  yellow_seconds
        self.ardinuo_second_error = .5
        self.agent=agent
        self.broker = broker
        self.global_message = status_free
        self.is_rl = True # if not rl action ,not stop it's real action
        self.topic_msg=MQTT_PARTOPIC_state+"/"+self.agent+"/"+MQTT_SUBTOPIC_msg
        self.topic_rep=MQTT_PARTOPIC_state+"/"+self.agent+"/"+MQTT_SUBTOPIC_reply
        self.initialize()
        # intiail video step 
        self._ret, self._frame_vid = self.lane_video_estimatior.cap.read() 
        if not self._ret or self._frame_vid is None:
            print("Warning: Failed to read initial frame from video")
            return
        self._estimate = self.lane_video_estimatior.speedestimator(self._frame_vid)
        self._last_real_action = 'g'

    def initialize(self):
        self.ardinuo = ardinuo(self.lane_id,self.ardinuo_port)
        self.lane_video_estimatior = Video_Estimatior(self.lane_id,self.global_url,self.yolo_path,self.thresould_speed,self.one_frame_processing)
        self._prepare_and_start_mqtt()

    def getTime(self):
        return 0

    
    def close(self):
        """
        Closes the connection.
        """
        #self.lane_video_estimatior.close_vid()
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

    # the next 2 fuctions are just to apply signal for protype ,not have real meaning out of prototype (same for way update video work)
    def pause(self):
        requests.post(self.global_url+"/"+route_pause)
    def resume(self):
        requests.post(self.global_url+"/"+route_resume)

    def read_video(self,action,duration):
            if (action == 'g'):

                self.resume()
                start_time = time.time()
                end_time = start_time + duration 
                while True:
                    elapsed = time.time() - start_time

                    if (time.time()  >= end_time - self.lane_video_estimatior.one_frame_processing):#- conn.one_frame_processing you can remove it ,but just for accurate results
                        break 
                    try:
                        self._ret,self._frame = self.lane_video_estimatior.cap.read()
                        if not self._ret or self._frame is None:
                            print("Warning: Failed to read frame")
                            return
                    except cv2.error as e:
                        print(f"OpenCV error: {e}")
                        raise
                        
                    except Exception as e:
                        print(f"Unexpected error: {e}")
                        raise
                    

                    self._estimate = self.lane_video_estimatior.speedestimator(self._frame)
                    if (self.global_message != status_free) and self.is_rl: #if not rl action ,not stop it's real action
                        return
                    

                # Ensure the total execution time is exactly `duration`
                remaining_time = end_time - time.time()
                if remaining_time > 0:
                    time.sleep(remaining_time)
                

            else:
                self.pause()
                start_time = time.time()  
                while True:
                    #print(global_message)
                    if (self.global_message != status_free) and self.is_rl: #if not rl action ,not stop it's real action
                        return
                    elapsed = time.time() - start_time
                    if elapsed > duration:
                        break

    def do_steps_duration(self, action_lane_i,duration, max_sumo_step, agent, traffic_scale):
        """
        Performs simulation steps for a given duration.
        """
        duration_wanted_without_ardinuo = duration - self.ardinuo_second_error
        self.ardinuo.send_command(action_lane_i,duration-1) # duration-1 as it consider zero
        self.read_video(action_lane_i,duration_wanted_without_ardinuo) 
        

    

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

    def _final_order_message(self,message):
        if message==status_free:
            return True
        
        parts = message.split()
        if len(parts) == 2 and parts[0] in [status_acc, status_emr] and parts[1].isdigit():
            return True
        
        return False

    



    def _prepare_and_start_mqtt(self):
        self._prepare_listener()
        self._prepare_publisher()
        pass

    def _prepare_publisher(self):
        def mqtt_listener():
            def on_message(client, userdata, message):
                global global_message
                message = str (message.payload.decode())
                print(f"📥 Received: {message}")
                if (message =="CHK"):
                    client.publish(self.topic_rep, "AVBL",retain=True, qos=1)

                if ('QL' in message):
                    client.publish(self.topic_rep, 'QL 10',retain=True, qos=1)
                if  self._final_order_message(message):
                    self.global_message = message

        

            client = mqtt.Client()
            client.connect(self.broker, 1883)
            client.subscribe(self.topic_msg)
            client.on_message = on_message
            client.loop_forever()  # هيفضل شغال في الخلفية

        # شغل الـ MQTT client في Thread منفصل عشان ما يعلقش البرنامج
        mqtt_thread = threading.Thread(target=mqtt_listener, daemon=True)
        print(f"------------ STARTING MQTT PUBLISHER AND LISTENER AT {self.topic_msg}------------")
        mqtt_thread.start()
        

    def _prepare_listener(self):
         
        def mqtt_listener():
            def on_message(client, userdata, message):

                print(f"📥 Received: {message.payload.decode()}")

            client = mqtt.Client()
            client.connect(self.broker, 1883)
            client.subscribe(self.topic_rep)
            client.on_message = on_message
            client.loop_forever()  # هيفضل شغال في الخلفية

            # شغل الـ MQTT client في Thread منفصل عشان ما يعلقش البرنامج
        mqtt_thread = threading.Thread(target=mqtt_listener, daemon=True)
        print(f"------------ STARTING MQTT LISTENER AT {self.topic_rep} ------------")
        mqtt_thread.start()

