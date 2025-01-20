import cv2 
import mediapipe as mp #for detecting my body
import numpy as np #for math calculation of my body angles

# Initialize Mediapipe Pose Detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Second point (vertex)
    c = np.array(c)  # Third point
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Initialize variables for counting repetitions
counter = 0
stage = None  # "down" or "up"

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Detect pose
    results = pose.process(image)

    # Convert back to BGR for OpenCV
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Extract landmarks
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Get coordinates for hips and knees (example for squats)
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, 
                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        # Calculate angle
        angle = calculate_angle(hip, knee, ankle)

        # Visualize angle on the video
        cv2.putText(image, str(int(angle)), 
                    tuple(np.multiply(knee, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                    )

        # Squat logic
        if angle > 160:  # Leg fully extended
            stage = "up"
        if angle < 90 and stage == "up":  # Squat position
            stage = "down"
            counter += 1

        # Display counter
        cv2.putText(image, f"Reps: {counter}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Render pose landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Show video feed
    cv2.imshow('Workout Repetition Counter', image)

    # Break loop on 'q' key press
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
