import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Define constants for the left and right eye landmarks
LEFT_EYE = [33, 133, 160, 159, 158, 153, 144, 163]
RIGHT_EYE = [362, 263, 249, 390, 373, 374, 380, 385]

# Define iris indices (based on Mediapipe's face mesh)
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Function to calculate iris visibility
def get_iris_visibility(landmarks, iris_indices):
    try:
        # Extract the x, y, z values of the landmarks and calculate the mean position
        iris_points = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in iris_indices if i < len(landmarks)])
        if len(iris_points) < 4:
            return None  # Return None if not enough iris points are detected

        iris_center = np.mean(iris_points, axis=0)  # Calculate the center of the iris

        # Calculate the positions for the left and right eye corners (left_eye and right_eye)
        eye_left = np.array([landmarks[LEFT_EYE[0]].x, landmarks[LEFT_EYE[0]].y, landmarks[LEFT_EYE[0]].z])
        eye_right = np.array([landmarks[RIGHT_EYE[3]].x, landmarks[RIGHT_EYE[3]].y, landmarks[RIGHT_EYE[3]].z])

        # Calculate the eye width as the Euclidean distance between the left and right eye corners
        eye_width = np.linalg.norm(eye_left - eye_right)

        # Calculate the distance from the iris center to the eye corners
        iris_distance = np.linalg.norm(iris_center - eye_left) + np.linalg.norm(iris_center - eye_right)

        # Calculate iris visibility as a ratio of iris distance to eye width
        iris_visibility = iris_distance / eye_width
        return iris_visibility
    except IndexError:
        return None  # Return None if there is an index error


# Function to check sleep or drowsiness
def check_sleep(iris_visibility):
    if iris_visibility is not None and iris_visibility > 1.2:
        return True  # Sleep detected (eyes are mostly closed)
    else:
        return False  # No sleep detected


# Initialize webcam and face mesh model
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame for better visualization
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw the landmarks
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

                # Get iris visibility for both eyes
                left_iris_visibility = get_iris_visibility(face_landmarks.landmark, LEFT_IRIS)
                right_iris_visibility = get_iris_visibility(face_landmarks.landmark, RIGHT_IRIS)

                # Check if sleep is detected
                if check_sleep(left_iris_visibility) or check_sleep(right_iris_visibility):
                    cv2.putText(frame, "Sleep Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "No Sleep Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Sleep Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
