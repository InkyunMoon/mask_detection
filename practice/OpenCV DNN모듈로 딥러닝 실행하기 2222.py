'''
https://wiserloner.tistory.com/892
OpenCV에서는 3.1버전부터 딥러닝 기술을 사용할 수 있는 dnn모듈을 제공한다. OpenCV는 딥러닝 전문 라이브러리는 아니지만 영상에 딥러닝 기능을 추가시켜주는 용도로 활용 가능하다.
기존의 텐서플로우, 다크넷 등 다른 딥러닝 프레임워크에서 모델을 학습하고, 학습된 모델을 가져와서 '순전파'시키는 역할을 한다.

- OpenCV의 딥러닝 네트워크는 cv::dnn::Net클래스를 이용해서 표현한다. Net클래스 객체는 보통 사용자가 직접 생성하지 않고, readNet()함수를 사용하여 모델을 가져오는 개념이다.
Net을 불러오는게 OpenCV 딥러닝 모듈의 전부라고 할 수 있다. 따라서, 제대로 객체가 설정되었는지 알아보고 예외처리를 하는 것이 좋다.
- Net 객체를 만들었다면, 영상을 입력하고 영상에 대한 딥러닝 분석 결과를 받는 단계만이 남았다. 딥러닝 모듈을 실행시키려면 blob이란 형태로 영상을 변환시킨 후, 이것을 활용한다.
blob은 영상 데이터를 NCHW정보로 표현한다.(N-영상개수, C-채널개수, H-height, W-width) 이 정보가 있어야 딥러닝 모듈이 데이터를 처리할 수 있다.

- 입력데이터가 세팅되었다면, 그대로 순전파를 시킨다. 순전파란, 신경망의 회귀 분류에 대한 예측을 실행한다는 의미이다. 학습된 모델을 사용하여 탐지된 객체에 대한 라벨을 리턴한다.

https://deep-learning-study.tistory.com/299
1. 네트워크 불러오기 - cv2.dnn.readNet
- opencv로 딥러닝 실행하기 위해서 dnn net 클래스 객체 생성 필요
- cfg, weights파일 필요

2. 네트워크 blob 만들기 - cv2.dnn.blobFromImage
- 입력 영상을 블롭 객체로 만들어서 추론 진행
- 학습된 모델이 어떻게 학습되었는지 파악하고 그에 맞게 인자를 입력
- 하나의 영상 추론시 cv2.dnn.blobFromImage, 여러 영상 추론시 cv2.dnn.blobFromImages 메서드 사용

3. 네트워크 입력 설정하기 - cv2.dnn_Net.setInput

4. 네트워크 순방향 실행(inference) - cv2.dnn_Net.Forward
- 네트워크를 어떻게 생성했냐에 따라 출력을 여러개 지정할 수 있다.
'''
import numpy as np
import cv2

# YOLO 로드
net = cv2.dnn.readNet("/home/piai/Documents/darknet/weights/yolov3.weights", "/home/piai/Documents/darknet/cfg/yolov3.cfg")
classes = []
with open("/home/piai/Documents/darknet/data/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layers_names = net.getLayerNames()
# outputlayer를 얻고자 한다. 이것을 통해 객체를 탐지할 수 있을 것이다
output_layers = [layers_names[i[0] -1] for i in net.getUncoqnnectedOutLayers()]
colors = np.random.uniform(0, 255, size = (len(classes), 3))

img = cv2.VideoCapture(0)

# 이미지 로딩하기
# img = cv2.imread("/home/piai/Documents/darknet/data/dog.jpg")
img = cv2.resize(img, None, fx = 0.4, fy = 0.4)
height, width, channels = img.shape

# detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop = False)
net.setInput(blob) # 입력한 데이터에 대해서 예측을 실행한다..?

outs = net.forward(output_layers) # outs에 탐지된 객체의 모든 정보가 배열로 담긴다. 위치와 확률 등.
# 여기까지 객체 탐지가 완료되었다. 이제 탐지된 객체의 정보를 화면에 나타내주는 단계로 넘어가도록 한다.

# 정보 화면에 표시하기
class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence >= 0.5:   
            # 객체 탐지됨
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            
            # 사각형 좌표
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w ,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

            # # cv2.rectangle(img, (center_x, center_y), 10, (0, 255, 0), 2)
            # cv2.imshow("Image", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 2, color, 2)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()