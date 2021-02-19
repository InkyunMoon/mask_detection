# https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/
import cv2
import numpy as np
import time

# base = '/home/piai/Documents/darknet/'
# net = cv2.dnn.readNet(base+'/backup/yolo-obj_last.weights', base+'yolo-obj.cfg')

# classes = []

# with open(base+'data/obj.names', 'r') as f:
#     classes = [line.strip() for line in f.readlines()]

# layer_names = net.getLayerNames()
# output_layers = [layer_names[i[0] -1] for i in net.getUnconnectedOutLayers()]
# colors = np.random.uniform(0, 255, size=(len(classes),3))

'''
웹캠으로 촬영한 영상을 바로 네트워크에 사용할 수 없다.
blob으로 변환하여 피쳐를 추출하고 크기를 조정해야한다.
YOLO는 세가지 사이즈를 사용한다.
1. 320*320
2. 609*609
3. 416*416
'''
# cap = cv2.VideoCapture(0)

# while True:
#     success, img = cap.read()
#     # img = cv2.resize(img, None, fx=0.4, fy=0.4)
#     # height, width, channels = img.shape

#     # 읽어들인 Img를 Blob으로 변환
#     blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416),(0,0,0), True, crop = False)
#     net.setInput(blob)
#     outs = net.forward(output_layers)
#     height, width, channels = img.shape

#     class_ids = []
#     confidences = []
#     boxes = []
#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.7:
#                 # Object detected
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#                 # Rectangle coordinates
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)
    
#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#     font = cv2.FONT_HERSHEY_PLAIN
#     for i in range(len(boxes)):
#         if i in indexes:
#             x, y, w, h = boxes[i]
#             label = str(classes[class_ids[i]])
#             color = colors[i]
#             cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
#             cv2.putText(img, label, (x, y + 30), font, 2, color, 2)

#     cv2.imshow("Image", img)

#     if cv2.waitKey(1) == ord('q'):
#         cv2.destroyAllWindows()
#         break

#-----------------------------------------------------------------------
classes = []
base = '/home/piai/Documents/darknet/'
net = cv2.dnn.readNet(base+'/backup/yolo-obj_last.weights', base+'yolo-obj.cfg')
with open(base+'data/obj.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

def inference(image):
    ih, iw = image.shape[:2]
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    inference_time = end - start

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > 0.5:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([iw, ih, iw, ih])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    results = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            # extract the bounding box coordinates
            x, y = (boxes[i][0], boxes[i][1])
            w, h = (boxes[i][2], boxes[i][3])
            id = classIDs[i]
            confidence = confidences[i]

            results.append((id, classes, confidence, x, y, w, h))

    return iw, ih, inference_time, results

def mask_detection(frame):
    width, height, inference_time, results = inference(frame)
    for detection in results:
        id, name, confidence, x, y, w, h = detection
        cx = x + (w / 2)
        cy = y + (h / 2)

        # draw a bounding box rectangle and label on the image
        color = (0, 255, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = "%s (%s)" % (name, round(confidence, 2))
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)
        return frame

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    cv2.imshow('mask', mask_detection(frame))
    # cv2.imshow('mask', mask_detection(frame))

    key = cv2.waitKey(1) # 키 입력을 1ms동안 대기한다. 0을 인자로 설정할 시, 입력이 있을 때 까지 무한 대기한다.
    if key == 'q':
        break