# 코드 출처: https://pysource.com/2019/03/12/face-landmarks-detection-opencv-with-python/

import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0) # 0번 카메라를 사용하여 비디오를 캡쳐한다.(1프레임씩)

detector = dlib.get_frontal_face_detector() # 정면 얼굴을 탐지하도록 한다.
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # 얼굴에 68개의 랜드마크를 predict할 것이다.

while True:
    _, frame = cap.read() # cap.read()는 캡쳐한 프레임을 하나씩 읽어들인다.
    # ret, frame을 리턴하는데, ret는 프레임을 제대로 읽었는지 판단하는 Boolean값이며 읽은 프레임은 frame이다.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 읽어들인 frame을 grayscale로 변환한다. 데이터의 차원을 낮춤으로써 CPU의 부하를 방지한다. grayscale일지라도 원하는 정보는 모두 포함하고있다.

    faces = detector(gray)
    # face는 rectangle object인데, interative.
    # loop를 돌며 얼굴의 좌표를 가져올 수 있다.
    # face -> [(왼쪽 상단), (오른쪽 하단)]
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
         # frame에 처리를 할 수 있다. 위의 코드는 얼굴 주변에 초록색의 3의 두께를 가진 사각형을 씌워주는 작업이다.
        
        landmarks = predictor(gray, face)

        for n in range(0, 68):
            x = landmarks.part(n).x # 0~67의 인덱스 중 n번째 점의 x좌표를 가져온다.
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
            # (frame, (x, y), radius, (color), thickness)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) # 키 입력을 1ms동안 대기한다. 0을 인자로 설정할 시, 입력이 있을 때 까지 무한 대기한다.
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()