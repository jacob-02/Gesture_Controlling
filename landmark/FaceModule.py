import mediapipe as mp
import cv2


class FaceDetector:
    def __init__(self, mode=False, max_face=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.max_face = max_face
        self.trackCon = trackCon
        self.detectionCon = detectionCon
        self.mpFace = mp.solutions.face_mesh
        self.face = self.mpFace.FaceMesh(self.mode, self.max_face, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.detectedFace = False

    def findFace(self, frame, draw=True):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.face.process(image)

        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, faceLms, self.mpFace.FACE_CONNECTIONS,
                                               self.mpDraw.DrawingSpec(color=(0, 0, 0), thickness=1, circle_radius=1),
                                               self.mpDraw.DrawingSpec(color=(155, 155, 155), thickness=2,
                                                                       circle_radius=2))
        return frame

    def findPosition(self, frame, faceNo=0, draw=True):
        lmList = []
        if self.results.multi_face_landmarks:
            faceLms = self.results.multi_face_landmarks[faceNo]
            self.detectedFace = True
            for id, lm in enumerate(faceLms.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 0), cv2.FILLED)

        else:
            self.detectedFace = False

        return lmList
