from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, PlainTextResponse
import cv2
import time
import threading

app = FastAPI()
video = cv2.VideoCapture("vid_2.mp4", cv2.CAP_FFMPEG)
paused = False
import threading
lock = threading.Lock()

def generate_frames():
    global paused, video, lock
    while True:
        if paused:
            time.sleep(0.02)
            continue

        with lock:
            success, frame = video.read()

        if not success:
            with lock:
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            time.sleep(0.05)
            continue

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.post("/pause")
async def pause_video():
    global paused
    paused = True
    return PlainTextResponse("Video paused")

@app.post("/resume")
async def resume_video():
    global paused
    paused = False
    return PlainTextResponse("Video resumed")
