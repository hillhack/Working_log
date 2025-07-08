import mediapipe as mp
from utils.metric_utils import calculate_ear
import cv2
class EyeTracker:
    def __init__(self, ear_threshold=0.25):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.EAR_THRESHOLD = ear_threshold
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        
    def detect_eye_state(self, frame):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_ear = calculate_ear(self.LEFT_EYE, landmarks, w, h)
            right_ear = calculate_ear(self.RIGHT_EYE, landmarks, w, h)
            avg_ear = (left_ear + right_ear) / 2.0
            eyes_closed = avg_ear < self.EAR_THRESHOLD
            return "Eyes Closed" if eyes_closed else "Eyes Open", round(avg_ear, 2)
        return "Unknown", 0.0