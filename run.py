import cv2
import time
import math
from subprocess import call
from landmark import HandModule

wCam, hCam = 700, 500

capture = cv2.VideoCapture(0)
capture.set(3, wCam)
capture.set(4, hCam)
count = 0
pTime = 0
volume = 20.0
volumeList = []

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

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        distance = math.hypot(x2 - x1, y2 - y1)

        volume = (distance - 15.0) // 3.25

        cv2.circle(frame, (x1, y1), 10, (0, 255, 0), cv2.FILLED)
        cv2.circle(frame, (x2, y2), 10, (0, 255, 0), cv2.FILLED)

        cv2.circle(frame, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

        if volume >= 100.0:
            volume = 100.0

        if volume <= 0:
            volume = 0

        cv2.putText(frame, str(int(volume)) + "%", (450, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(frame, str(int(fps)) + " fps", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)

    volumeList.append(volume)

    call(["amixer", "-D", "pulse", "sset", "Master", str(volume) + "%"])

    if not detector.detectedHand:
        cv2.putText(frame, 'No Hand Detected', (40, 450), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
        call(["amixer", "-D", "pulse", "sset", "Master", str(volumeList[0]) + "%"])
        count += 1
        if count == 100:
            break

    if detector.detectedHand:
        count = 0

    cv2.imshow('Webcam', frame)

    if cv2.waitKey(20) & 0xFF == ord('d'):
        call(["amixer", "-D", "pulse", "sset", "Master", str(volumeList[0]) + "%"])
        break

    volumeList.pop()

capture.release()
cv2.destroyAllWindows()
