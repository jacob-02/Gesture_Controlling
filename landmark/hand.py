import time
import mediapipe as mp
import cv2


def hand_detection():
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    pTime = 0
    cTime = 0

    capture = cv2.VideoCapture(0)

    while True:
        ret, frame = capture.read()
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    print(id, cx, cy)

                mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS,
                                      mpDraw.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
                                      mpDraw.DrawingSpec(color=(155, 155, 155), thickness=2,
                                                         circle_radius=2))

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(frame, str(int(fps)) + " fps", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)

        cv2.imshow('Webcam', frame)

        if cv2.waitKey(20) & 0xFF == ord('d'):
            break

    capture.release()
    cv2.destroyAllWindows()
