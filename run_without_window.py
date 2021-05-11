import cv2
import math
from subprocess import call
from landmark import HandModule

wCam, hCam = 700, 500

capture = cv2.VideoCapture(0)
capture.set(3, wCam)
capture.set(4, hCam)
pTime = 0
volume = 20.0
volumeList = []
muter = 20.0

detector = HandModule.HandDetector(detectionCon=0.8)

while True:
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)
    frame = detector.findHands(frame)

    lm = detector.findPosition(frame, draw=False)

    distance = 0

    if len(lm) != 0:
        x1, y1 = lm[4][1], lm[4][2]
        x2, y2 = lm[8][1], lm[8][2]
        x3, y3 = lm[20][1], lm[20][2]

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        distance = math.hypot(x2 - x1, y2 - y1)
        muter = math.hypot(x3 - x1, y3 - y1)

        volume = (distance - 15.0) // 3.25

        if volume >= 100.0:
            volume = 100.0

        if volume <= 0:
            volume = 0

    volumeList.append(volume)

    if muter > 40.0:
        call(["amixer", "-D", "pulse", "sset", "Master", str(volume) + "%"])

    if muter <= 40.0:
        call(["amixer", "-D", "pulse", "sset", "Master", str(0) + "%"])

    if cv2.waitKey(20) & 0xFF == ord('d'):
        call(["amixer", "-D", "pulse", "sset", "Master", str(volumeList[0]) + "%"])
        break

    volumeList.pop()

capture.release()
cv2.destroyAllWindows()
