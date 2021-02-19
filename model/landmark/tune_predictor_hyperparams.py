# import the necessary packages
import config
from sklearn.model_selection import ParameterGrid # iterable한 파라미터 조합을 생성
import multiprocessing
import numpy as np
import random
import time
import dlib
import cv2
import os

def evaluate_model_acc(xmlPath, predPath): # 모델 정확도를 측정 (Mean Average Error, MAE)
	# compute and return the error (lower is better) of the shape
	# predictor over our testing path
	return dlib.test_shape_predictor(xmlPath, predPath)
	
def evaluate_model_speed(predictor, imagePath, tests=10): # 모델 속도를 측정
	# initialize the list of timings
	timings = []
	
	# loop over the number of speed tests to perform
	for i in range(0, tests):
		# load the input image and convert it to grayscale
		image = cv2.imread(config.IMAGE_PATH)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 이미지를 불러와 그레이스케일 변환
		
		# detect faces in the grayscale frame
		detector = dlib.get_frontal_face_detector() 
		rects = detector(gray, 1) # HOG + Linear SVM face detector -> 얼굴 탐지
		
		# ensure at least one face was detected
		if len(rects) > 0: # 하나 이상의 얼굴이 탐지된다면
			# time how long it takes to perform shape prediction
			# using the current shape prediction model
			start = time.time() 
			shape = predictor(gray, rects[0])
			end = time.time() # 현재 예측 모델이 예측에 얼마나 시간이 걸리는지 측정
			
		# update our timings list
		timings.append(end - start) # 측정된 결과를 timings 리스트에 포함시킨다.
		
	# compute and return the average over the timings
	return np.average(timings) # 평균값으로 리턴
	
# define the columns of our output CSV file
cols = [
	"tree_depth", # 보통 2~8사이 값을 사용. regression tree의 깊이를 조절(ensemble of regression trees, ERT). 작은 값: 빠르지만 부정, 큰 값: 느리지만 정확.
	"nu", # 0~1사이 값을 사용. 1에 가까운 값: 훈련 셋에 가깝게 피팅. 0에 가까운 값: 일반화시키고자 하나 0에 가까울수록 훈련 데이터가 더 필요하다.
	"cascade_depth", # ERT로부터 초기 예측값들을 튜닝하는데 사용. 정확도와 모델 크기에 큰 영향을 미친다. 큰 값: 모델사이즈 크고 더 정확. 작은 값: 모델사이즈 작고 부정확. 일반적으로 6~18 사이의 값을 사용
	"feature_pool_size", # 각각의 cascade에서 랜덤트리를 일반화하는데 사용될 픽셀의 수. 큰 값: 더 정확, 느림. 속도가 중요하지 않으면 큰 값을 사용한다. 작은 값은 자원이 제한된 환경에서 사용할 때 사용.
	"num_test_splits", # 모델 학습시간에 큰 영향을 미친다. 큰 값: 정확한 성능, 그러나 학습 오래걸림
	"oversampling_amount", # 학습에 사용되는 데이터 augmentation을 조절. 인풋 이미지에 대해 N개의 랜덤 변형을 적용하는 것. Regularization parameter로 사용할 수 있다. 학습시간에 영향을 미친다.
	"oversampling_translation_jitter", # 일반적으로 0~0.5사이의 값 사용. 학습 데이터셋에 적용되는 translation augmentation의 양을 조절
	"inference_speed",
	"training_time",
	"training_error",
	"testing_error",
	"model_size"
]

# open the CSV file for writing and then write the columns as the
# header of the CSV file
csv = open(config.CSV_PATH, "w")
csv.write("{}\n".format(",".join(cols)))

# determine the number of processes/threads to use
procs = multiprocessing.cpu_count()
procs = config.PROCS if config.PROCS > 0 else procs

# initialize the list of dlib shape predictor hyperparameters that
# we'll be tuning over
hyperparams = {
	"tree_depth": list(range(4, 10, 2)),
	"nu": [0.2, 0.3, 0.4],
	"cascade_depth": list(range(12, 20, 2)),
	"feature_pool_size": [1000],
	"num_test_splits": [300],
	"oversampling_amount": [40],
	"oversampling_translation_jitter": [0.1, 0.25]
}

# construct the set of hyperparameter combinations and randomly
# sample them as trying to test *all* of them would be
# computationally prohibitive
combos = list(ParameterGrid(hyperparams))
random.shuffle(combos)
sampledCombos = combos[:config.MAX_TRIALS]
print("[INFO] sampling {} of {} possible combinations".format(
	len(sampledCombos), len(combos)))
	
	# loop over our hyperparameter combinations
for (i, p) in enumerate(sampledCombos):
	# log experiment number
	print("[INFO] starting trial {}/{}...".format(i + 1,
		len(sampledCombos)))
	
	# grab the default options for dlib's shape predictor and then
	# set the values based on our current hyperparameter values
	options = dlib.shape_predictor_training_options()
	options.tree_depth = p["tree_depth"]
	options.nu = p["nu"]
	options.cascade_depth = p["cascade_depth"]
	options.feature_pool_size = p["feature_pool_size"]
	options.num_test_splits = p["num_test_splits"]
	options.oversampling_amount = p["oversampling_amount"]
	otj = p["oversampling_translation_jitter"]
	options.oversampling_translation_jitter = otj
	# tell dlib to be verbose when training and utilize our supplied
	# number of threads when training
	options.be_verbose = True
	options.num_threads = procs


	# train the model using the current set of hyperparameters
	start = time.time()
	dlib.train_shape_predictor(config.TRAIN_PATH,
		config.TEMP_MODEL_PATH, options)
	trainingTime = time.time() - start
	
	# evaluate the model on both the training and testing split
	trainingError = evaluate_model_acc(config.TRAIN_PATH,
		config.TEMP_MODEL_PATH)
	testingError = evaluate_model_acc(config.TEST_PATH,
		config.TEMP_MODEL_PATH)
		
	# compute an approximate inference speed using the trained shape
	# predictor
	
	predictor = dlib.shape_predictor(config.TEMP_MODEL_PATH)
	inferenceSpeed = evaluate_model_speed(predictor,
		config.IMAGE_PATH)
		
	# determine the model size
	modelSize = os.path.getsize(config.TEMP_MODEL_PATH)
	# build the row of data that will be written to our CSV file
	row = [
		p["tree_depth"],
		p["nu"],
		p["cascade_depth"],
		p["feature_pool_size"],
		p["num_test_splits"],
		p["oversampling_amount"],
		p["oversampling_translation_jitter"],
		inferenceSpeed,
		trainingTime,
		trainingError,
		testingError,
		modelSize,
	]
	row = [str(x) for x in row]
	# write the output row to our CSV file
	csv.write("{}\n".format(",".join(row)))
	csv.flush()
	# delete the temporary shape predictor model
	if os.path.exists(config.TEMP_MODEL_PATH):
		os.remove(config.TEMP_MODEL_PATH)
# close the output CSV file
print("[INFO] cleaning up...")
csv.close()

