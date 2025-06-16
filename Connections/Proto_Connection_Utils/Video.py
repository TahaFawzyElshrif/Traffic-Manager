import cv2
from ultralytics import solutions
import time
import logging
import numpy as np

# تعطيل الرسائل الخاصة بالمكتبة
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

class Video_Estimatior():
        


    fps = 0

    video_writer = None
    speedestimator = None
    screen = None
    cap = None
    waiting_time_full = {}

    def __init__(self,lane_id,global_url_,yolo_path_,thresould_speed_,one_frame_processing_,):
        self.yolo_path = yolo_path_
        self.global_url = global_url_
        self.thresould_speed = thresould_speed_
        self.one_frame_processing = one_frame_processing_
        self.lane_id=lane_id
        self.intialize_Model_CV()
        
        
    def intialize_Model_CV(self):
        self.cap = cv2.VideoCapture(self.global_url+"/video_feed")
        assert self.cap.isOpened(), "Error reading video file"
        w, h, self.fps = (int(self.cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        self.speedestimator = solutions.SpeedEstimator(
            show=False,
            model=self.yolo_path,
            fps=self.fps,
            conf=.25,
        )

        self.one_second_processing = self.fps * self.one_frame_processing



                



    def get_speeds(self):
        speed_vechiles_km_h = self.speedestimator.spd
        speed_vechiles_m_s = {key : (value *(5/18)) for key,value in speed_vechiles_km_h.items()}
        speeds = list(speed_vechiles_m_s.values())
        return (speeds)
        
    def get_waiting_video(self):
    
        AccumulatedWaitingTime = np.array(list(self.waiting_time_full.values()))

        return AccumulatedWaitingTime

    def update_waiting_red_video(self,duration):
        self.waiting_time_full = { key:(val+duration) for key,val in self.waiting_time_full.items()}

    def update_waiting_video(self):
        speed_vechiles_km_h = self.speedestimator.spd
        waiting_vehicles = { key:val for key,val in speed_vechiles_km_h.items() if ((5/18)*val)<self.thresould_speed}
        
        for key,val in waiting_vehicles.items() : 
            self.waiting_time_full[key] = self.waiting_time_full.get(key, 0) + (1 / self.fps)

        for key,val in self.waiting_time_full.items() : 
            if key not in waiting_vehicles.keys():
                self.waiting_time_full.pop(key)
        
    def get_State_video(self):
        
        speeds = self.get_speeds()

        
        waiting_time = self.get_waiting_video()
        
        mean_speed = np.mean(speeds) if (len(speeds)>0) else 0 
        variance_speed = np.var(speeds) if (len(speeds)>0) else 0 
        mean_waiting_time = np.mean(waiting_time) if (len(waiting_time)>0) else 0 
        variance_waiting_time = np.var(waiting_time) if (len(waiting_time)>0) else 0

        ## Estimation ,future work more using
        occupancy = len(speeds) / 10  # assuming max 10 vehicles per lane
        occupancy = min(occupancy, 1.0)  * 100

        queue_length = len(waiting_time) # The number of vehicles waiting in the queue
        throughput = len(speeds) / (1 / self.fps) # The formula divides len(speeds) by (1 / self.fps), which is the same as multiplying by fps. This converts the count of vehicles into a rate of vehicles passing per time unit 
        
        return np.array([mean_speed,variance_speed,mean_waiting_time,variance_waiting_time,throughput,queue_length,occupancy])


