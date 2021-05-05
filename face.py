import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

capture = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        ret, frame = capture.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = holistic.process(image)
        print("Face Landmarks", results.face_landmarks)
        print("Left Hand Landmarks", results.left_hand_landmarks)
        print("Right Hand Landmarks", results.right_hand_landmarks)
        print("Pose Landmarks", results.pose_landmarks)

        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, mp_drawing.DrawingSpec(color=(200, 0, 10), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(0, 220, 0), thickness=2, circle_radius=2))  # This is for face
        # landmarks
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(200, 0, 10), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(0, 220, 0), thickness=2, circle_radius=2))  # This is for
        # left hand landmarks
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(200, 0, 10), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(0, 220, 0), thickness=2, circle_radius=2))  # This is for
        # right hand landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(200, 0, 10), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(0, 220, 0), thickness=2, circle_radius=2))  # This is for pose
        # landmarks

        cv2.imshow('Webcam', image)

        if cv2.waitKey(20) & 0xFF == ord('d'):
            break

capture.release()
cv2.destroyAllWindows()