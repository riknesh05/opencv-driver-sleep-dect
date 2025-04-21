import cv2
import mediapipe as mp
import numpy as np

# Eye indices for MediaPipe Face Mesh (right and left eye landmarks)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Function to calculate Eye Aspect Ratio (EAR)
def get_ear(landmarks, eye_indices):
    p = [landmarks[i] for i in eye_indices]
    vertical_1 = np.linalg.norm(np.array([p[1].x, p[1].y]) - np.array([p[5].x, p[5].y]))
    vertical_2 = np.linalg.norm(np.array([p[2].x, p[2].y]) - np.array([p[4].x, p[4].y]))
    horizontal = np.linalg.norm(np.array([p[0].x, p[0].y]) - np.array([p[3].x, p[3].y]))
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

# Webcam access
cap = cv2.VideoCapture(0)

# Drowsiness settings
EAR_THRESHOLD = 0.25
CLOSED_FRAMES_THRESHOLD = 50
closed_frames = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        left_ear = get_ear(landmarks, LEFT_EYE)
        right_ear = get_ear(landmarks, RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < EAR_THRESHOLD:
            closed_frames += 1
        else:
            closed_frames = 0

        if closed_frames > CLOSED_FRAMES_THRESHOLD:
            cv2.putText(frame, "DROWSINESS DETECTED!", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # Draw eyes
        for i in LEFT_EYE + RIGHT_EYE:
            x = int(landmarks[i].x * w)
            y = int(landmarks[i].y * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow("Drowsiness Detector", frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
