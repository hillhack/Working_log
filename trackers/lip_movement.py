import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
from utils.metric_utils import calculate_mar

class LipMovementDetector:
    def __init__(self, mar_threshold=0.04, window_size=5, talking_timeout=1.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.MAR_THRESHOLD = mar_threshold
        self.lip_distances = deque(maxlen=window_size)
        self.LIP_POINTS = [13, 14, 78, 308]
        self.last_talking_time = 0
        self.is_talking = False
        self.prev_is_talking = False  # Track previous state
        self.talking_timeout = talking_timeout
        
    def detect_talking(self, frame):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        current_time = time.time()
        mar = 0.0
        state_change = False  # Track if state changed from not talking to talking

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            mar = calculate_mar(self.LIP_POINTS, landmarks, w, h)
            self.lip_distances.append(mar)

            if len(self.lip_distances) == self.lip_distances.maxlen:
                std_dev = np.std(self.lip_distances)
                
                # Detect state change
                if std_dev > self.MAR_THRESHOLD:
                    self.last_talking_time = current_time
                    if not self.is_talking:
                        state_change = True  # State changed to talking
                    self.is_talking = True
                elif current_time - self.last_talking_time > self.talking_timeout:
                    self.is_talking = False
                    
                # Detect state transition (not talking -> talking)
                if not self.prev_is_talking and self.is_talking:
                    state_change = True
                    
                self.prev_is_talking = self.is_talking

        status = "Talking" if self.is_talking else "Not Talking"
        return status, round(mar, 3), state_change