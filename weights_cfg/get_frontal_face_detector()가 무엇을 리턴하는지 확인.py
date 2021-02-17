from imutils.video import VideoStream
import dlib
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2

path = '/home/piai/github/Learning_OpenCV/'

# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
# args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('/home/piai/Documents/68_landmarks_temp/shape_predictor_68_face_landmarks.dat')

img = cv2.imread(path + 'Resources/lena.png')
frame = imutils.resize(img, width=400)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 0)

# shape = predictor(gray, rects)
# shape = face_utils.shape_to_np(shape)

r = rects[0]
cv2.rectangle(frame, (r.top(),r.left()), (r.bottom(),r.right()),(0,0,255), 2, cv2.FILLED)
cv2.imshow('frame', frame)

if cv2.waitKey(0) == ord("q"):
    cv2.destroyAllWindows()