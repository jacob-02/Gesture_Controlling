import cv2
import time
import math
from landmark import HandModule

wCam, hCam = 700, 500

capture = cv2.VideoCapture(0)
capture.set(3, wCam)
capture.set(4, hCam)
pTime = 0

detector = HandModule.HandDetector(detectionCon=0.8)

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volume.GetMute()
volume.GetMasterVolumeLevel()
volume.GetVolumeRange()
volume.SetMasterVolumeLevel(-20.0, None)

while True:
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)
    frame = detector.findHands(frame)

    lm = detector.findPosition(frame, draw=False)

    if len(lm) != 0:
        x1, y1 = lm[4][1], lm[4][2]
        x2, y2 = lm[8][1], lm[8][2]

        cx, cy = (x1+x2)//2, (y1 + y2)//2

        distance = math.hypot(x2 - x1, y2 - y1)

        cv2.circle(frame, (x1, y1), 10, (0, 255, 0), cv2.FILLED)
        cv2.circle(frame, (x2, y2), 10, (0, 255, 0), cv2.FILLED)

        if distance >= 30.0:
            cv2.circle(frame, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

        cv2.putText(frame, str(float(distance)) + " units", (10, 450), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(frame, str(int(fps)) + " fps", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)

    cv2.imshow('Webcam', frame)

    if cv2.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv2.destroyAllWindows()
