import cv2
import mediapipe as mp
import numpy as np
import datetime

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
HIGHLIGHT_POINTS = {
    "left": [385, 380],
    "right": [160, 144]
}

EAR_THRESHOLD = 0.25
CLOSED_FRAMES_THRESHOLD = 50
closed_frames = 0

# Logging function
def log_drowsiness(reason):
    with open("drowsiness_log.txt", "a") as f:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{now}] DROWSINESS DETECTED - Reason: {reason}\n")

# EAR Calculation
def get_ear(landmarks, eye_indices):
    p = [landmarks[i] for i in eye_indices]
    vertical_1 = np.linalg.norm(np.array([p[1].x, p[1].y]) - np.array([p[5].x, p[5].y]))
    vertical_2 = np.linalg.norm(np.array([p[2].x, p[2].y]) - np.array([p[4].x, p[4].y]))
    horizontal = np.linalg.norm(np.array([p[0].x, p[0].y]) - np.array([p[3].x, p[3].y]))
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

# MediaPipe Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    status_text = ""

    if not results.multi_face_landmarks:
        status_text = "Head Down / Face Not Detected"
        log_drowsiness(status_text)
    else:
        landmarks = results.multi_face_landmarks[0].landmark
        left_ear = get_ear(landmarks, LEFT_EYE)
        right_ear = get_ear(landmarks, RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0

        # Check for visibility of eyes (very rough condition — can be customized)
        eye_visible = all(
            0 <= landmarks[i].x <= 1 and 0 <= landmarks[i].y <= 1
            for i in HIGHLIGHT_POINTS["left"] + HIGHLIGHT_POINTS["right"]
        )

        if not eye_visible:
            status_text = "Eyes Not Visible"
            log_drowsiness(status_text)

        elif avg_ear < EAR_THRESHOLD:
            closed_frames += 1
            if closed_frames > CLOSED_FRAMES_THRESHOLD:
                status_text = "Eyes Closed"
                log_drowsiness(status_text)
        else:
            closed_frames = 0

        # Draw eyes
        for i in LEFT_EYE + RIGHT_EYE:
            x = int(landmarks[i].x * w)
            y = int(landmarks[i].y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Draw specific points
        for eye in HIGHLIGHT_POINTS:
            for i in HIGHLIGHT_POINTS[eye]:
                x = int(landmarks[i].x * w)
                y = int(landmarks[i].y * h)
                cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)

    if status_text:
        cv2.putText(frame, f"DROWSINESS: {status_text}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
