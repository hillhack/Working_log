import numpy as np

def calculate_ear(eye_indices, landmarks, w, h):
    points = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in eye_indices]
    p1, p2, p3, p4, p5, p6 = points
    vertical1 = np.linalg.norm(p2 - p6)
    vertical2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

def calculate_mar(lip_indices, landmarks, w, h):
    top = np.array([landmarks[lip_indices[0]].x * w, landmarks[lip_indices[0]].y * h])
    bottom = np.array([landmarks[lip_indices[1]].x * w, landmarks[lip_indices[1]].y * h])
    left = np.array([landmarks[lip_indices[2]].x * w, landmarks[lip_indices[2]].y * h])
    right = np.array([landmarks[lip_indices[3]].x * w, landmarks[lip_indices[3]].y * h])
    
    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)
    mar = vertical / horizontal
    return mar