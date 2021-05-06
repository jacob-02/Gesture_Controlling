import mediapipe as mp
import cv2


def pose_detection():
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    capture = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = capture.read()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)

            results = holistic.process(image)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(155, 155, 155), thickness=2,
                                                             circle_radius=2))  # This is for pose
            # landmarks

            cv2.imshow('Webcam', image)

            if cv2.waitKey(20) & 0xFF == ord('d'):
                break

    capture.release()
    cv2.destroyAllWindows()
