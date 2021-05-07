import mediapipe as mp
import cv2


class HandDetector:
    def __init__(self, mode=False, max_hands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.trackCon = trackCon
        self.detectionCon = detectionCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.detectedHand = False

    def findHands(self, frame, draw=True):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS,
                                               self.mpDraw.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
                                               self.mpDraw.DrawingSpec(color=(155, 155, 155), thickness=2,
                                                                       circle_radius=2))
        return frame

    def findPosition(self, frame, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            handLms = self.results.multi_hand_landmarks[handNo]
            self.detectedHand = True
            for id, lm in enumerate(handLms.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 0), cv2.FILLED)

        else:
            self.detectedHand = False

        return lmList
