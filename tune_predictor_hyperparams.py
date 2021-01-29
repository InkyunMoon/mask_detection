# import the necessary packages
from pyimagesearch import config
from sklearn.model_selection import ParameterGrid
import multiprocessing
import numpy as np
import random
import time
import dlib
import cv2
import os

def evaluate_model_acc(xmlPath, predPath):
	# compute and return the error (lower is better) of the shape
	# predictor over our testing path
	return dlib.test_shape_predictor(xmlPath, predPath)
	
def evaluate_model_speed(predictor, imagePath, tests=10):
	# initialize the list of timings
	timings = []
	
	# loop over the number of speed tests to perform
	for i in range(0, tests):
		# load the input image and convert it to grayscale
		image = cv2.imread(config.IMAGE_PATH)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		
		# detect faces in the grayscale frame
		detector = dlib.get_frontal_face_detector() # HOG + Linear SVM face detector
		rects = detector(gray, 1)
		
		# ensure at least one face was detected
		if len(rects) > 0:
			# time how long it takes to perform shape prediction
			# using the current shape prediction model
			start = time.time()
			shape = predictor(gray, rects[0])
			end = time.time()
			
		# update our timings list
		timings.append(end - start)
		
	# compute and return the average over the timings
	return np.average(timings)
	
# define the columns of our output CSV file
cols = [
	"tree_depth",
	"nu",
	"cascade_depth",
	"feature_pool_size",
	"num_test_splits",
	"oversampling_amount",
	"oversampling_translation_jitter",
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
