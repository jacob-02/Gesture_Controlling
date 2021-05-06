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
