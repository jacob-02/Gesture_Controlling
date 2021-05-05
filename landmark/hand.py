import mediapipe as mp
import cv2


def hand_detection():
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

            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=2,
                                                             circle_radius=2))  # This is for hand
            # landmarks

            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=2,
                                                             circle_radius=2))  # This is for hand
            # landmarks

            cv2.imshow('Webcam', image)

            if cv2.waitKey(20) & 0xFF == ord('d'):
                break

    capture.release()
    cv2.destroyAllWindows()
