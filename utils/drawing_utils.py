import cv2
import mediapipe as mp

def draw_face_annotations(frame, landmarks, eye_indices, lip_indices):
    h, w, _ = frame.shape
    mp.solutions.drawing_utils.draw_landmarks(
        frame,
        landmarks,
        mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
    )
    
    # Draw eye/lip boxes
    def draw_box(indices, color):
        pts = []
        for i in indices:
            lm = landmarks.landmark[i]
            pts.append((int(lm.x * w), int(lm.y * h)))
        if pts:
            x_min = min(p[0] for p in pts)
            x_max = max(p[0] for p in pts)
            y_min = min(p[1] for p in pts)
            y_max = max(p[1] for p in pts)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
    
    draw_box(eye_indices[0], (0, 255, 255))   # Left eye (yellow)
    draw_box(eye_indices[1], (0, 255, 255))   # Right eye (yellow)
    draw_box(lip_indices, (255, 0, 255))      # Lips (pink)