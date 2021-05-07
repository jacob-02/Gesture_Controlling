import cv2
import time
import math
from subprocess import call
from landmark import HandModule

wCam, hCam = 700, 500

capture = cv2.VideoCapture(0)
capture.set(3, wCam)
capture.set(4, hCam)
pTime = 0

detector = HandModule.HandDetector(detectionCon=0.8)

while True:
    ret, frame = capture.read()
    frame = detector.findHands(frame)

    lm = detector.findPosition(frame, draw=False)
    distance=0
    if len(lm) != 0:
        x1, y1 = lm[4][1], lm[4][2]
        x2, y2 = lm[8][1], lm[8][2]

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        distance = math.hypot(x2 - x1, y2 - y1)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    volume = distance / 2.2
    call(["amixer", "-D", "pulse", "sset", "Master", str(volume) + "%"])

    if cv2.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv2.destroyAllWindows()
