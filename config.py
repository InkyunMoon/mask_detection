# import the necessary packages
import os
# define the path to the training and testing XML files
TRAIN_PATH = os.path.join("/home/piai/Documents/ibug_300W_large_face_landmark_dataset",
	"labels_ibug_300W_train_eyes.xml")
TEST_PATH = os.path.join("/home/piai/Documents/ibug_300W_large_face_landmark_dataset",
	"labels_ibug_300W_test_eyes.xml")
	
	
# 하이퍼파라미터 튜닝에 상용될 임시 모델 파일 경로 - 앞서 만들었던 디폴트 모델
TEMP_MODEL_PATH = "temp.dat"

# define the path to the output CSV file containing the results of
# our experiments
# 각 튜닝 결과가 저장될 csv파일
CSV_PATH = "trials2.csv"

# define the path to the example image we'll be using to evaluate
# inference speed using the shape predictor
# 주어진 모델의 예측 속도를 평가에 사용될 이미지
IMAGE_PATH = "/home/piai/Documents/example_inkyun.jpg"

# define the number of threads/cores we'll be using when trianing our
# shape predictor models
PROCS = -1

# define the maximum number of trials we'll be performing when tuning
# our shape predictor hyperparameters
# 여러 조합에 걸쳐서 성능을 측정할 것 - 최대 조합의 개수를 아래의 인자로 설정하여 성능을 측정할 것이다.
MAX_TRIALS = 108


