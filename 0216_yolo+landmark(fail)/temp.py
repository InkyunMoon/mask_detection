import cv2
import numpy as np
import os
os.chdir('/home/piai/github/mask_detection')
from yolo_mask_detection import yolo_mask

def main():
    mask_class = yolo_mask.face_mask()

cap = cv2.VideoCapture(0)
while True:
    success, frame = cap.read()

    mask_class.mask_detection(frame)

    cv2.imshow('mask detection', frame)

    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break

if __main__ == '__main__':
    main()

