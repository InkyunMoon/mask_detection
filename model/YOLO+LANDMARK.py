from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os

# ap = argparse.ArgumentParser()
# ap.add_argument('-i', '--input', required=True, help='type of input video')
# ap.add_argument('-o', '--output', required=False, help='path to output video')
# ap.add_argument('-y', '--yolo', required=True, help='base path to YOLO DIRECTORY')
# ap.add_argument('-c', '--confidence', type=float, default=0.5, help='minimum probability to filter weak detections')
# ap.add_argument('-t', '--threshold', type=float, default=0.3, help='threshold when applying NMS')
# args = vars(ap.parse_args())

# temporary line
yolo_path = '/home/piai/Documents/darknet/model_yolo_mask'
lablsPath = os.path.sep.join([yolo_path,'obj.names'])
weightsPath = os.path.sep.join([yolo_path, 'yolo-obj_last.weights'])
configPath = os.path.sep.join([yolo_path, 'yolo-obj.cfg'])
LABELS = open('/home/piai/Documents/darknet/model_yolo_mask/obj.names').read().strip().split('\n')


# labelsPath = os.path.sep.join([args['yolo'], 'obj.names'])
# LABELS = open(labelsPath).read().strip().split('\n') # 위 라벨 데이터의 디렉토리를 입력하면 라벨만 담긴 리스트 리턴

# # 색상 선택
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS),3),dtype='uint8')
# # shape(size)가 (2,3)인 넘파이 배열을 생성하여 0~255까지의 랜덤 정수를 부여한다.
# weightPath = os.path.sep.join([args['yolo'], 'yolo-obj_last.weights'])
# configPath = os.path.sep.join([args['yolo'], 'yolo-obj.cfg'])

print('[INFO] loading YOLO from disk...')
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# net.getUnconnectedOutLayers() == array([[74]], dtype=int32)

predictor = dlib.shape_predictor('/home/piai/github/mask_detection/model/landmark/optimal_eye_predictor.dat')
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    height, width, channels = frame.shape
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame.shape == (480, 640, 3)
	
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (320,320), (0,0,0), True, crop=False)
    # blob.shape == (1, 3, 416, 416)
    net.setInput(blob)
    outs = net.forward(ln)
    # outs[0].shape == (845, 7)
    boxes = []
    confidences = []
    classIDs = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            #if confidence > args['confidence']:
            if confidence > 0.5:
                # box = detection[0:4] * np.array([W,H,W,H])
                # (centerX, centerY, width, height) = box.astype('int')

                # x = int(centerX - (width / 2))
                # y = int(centerY - (height / 2))
                # boxes.append([x, y, int(width), int(height)])
                # confidences.append(float(confidence))
                # classIDs.append(classID)
                centerX = int(detection[0] * width)
                centerY = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(centerX - w/2)
                y = int(centerY - h/2)
                #####################################
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                classIDs.append(classID)

    #idxs = cv2.dnn.NMSBoxes(boxes, confidences, args['confidence'], args['threshold'])
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
        # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            #shape = predictor(i)
            #shape = face_utils.shape_to_np(shape)
            #print(shape)

            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            for (sX, sY) in shape:
                cv2.circle(frame, (sX, sY), 1, (0, 0, 255), -1)

    cv2.imshow('video', frame)
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break

cap.release()
