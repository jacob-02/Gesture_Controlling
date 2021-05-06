import time
import mediapipe as mp
import cv2


def face_detection():
    mpFaces = mp.solutions.face_mesh
    faces = mpFaces.FaceMesh()
    mp_drawing = mp.solutions.drawing_utils

    pTime = 0
    cTime = 0

    capture = cv2.VideoCapture(0)

    while True:
        ret, frame = capture.read()
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = faces.process(image)

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                for id, lm in enumerate(faceLms.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    print(id, cx, cy)

                mp_drawing.draw_landmarks(frame, faceLms, mpFaces.FACE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                          mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1,
                                                                 circle_radius=1))

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(frame, str(int(fps)) + " fps", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)

        cv2.imshow('Webcam', frame)

        if cv2.waitKey(20) & 0xFF == ord('d'):
            break

    capture.release()
    cv2.destroyAllWindows()
