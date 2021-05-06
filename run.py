import cv2
import time
import numpy as np
from landmark import HandModule

wCam, hCam = 700, 500

capture = cv2.VideoCapture(0)
capture.set(3, wCam)
capture.set(4, hCam)
pTime = 0

detector = HandModule.HandDetector()

while True:
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)
    frame = detector.findHands(frame)

    lm = detector.findPosition(frame)
    if len(lm) != 0:
        print(lm[3])

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(frame, str(int(fps)) + " fps", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)

    cv2.imshow('Webcam', frame)

    if cv2.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv2.destroyAllWindows()
