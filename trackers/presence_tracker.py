# presence_tracker.py

import mediapipe as mp
import cv2
import csv
from datetime import datetime

LOG_FILE = 'data/presence_log.csv'

class PresenceTracker:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.6
        )
    
    def detect_presence(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        return results.detections is not None

# ✅ Add these two functions:
def init_log_file():
    try:
        with open(LOG_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Date', 'Start Time', 'End Time', 'Duration (seconds)', 'Eye Status', 'Talking'])
    except:
        pass

def log_presence(start, end, duration, eye_status, mouth_status):
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().date(), start, end, round(duration), eye_status, mouth_status])
