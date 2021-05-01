
import cv2
import pytesseract
import imutils
import numpy as np
import matplotlib.pyplot as plt
try:
 from PIL import Image
except ImportError:
 import Image
import numpy as np
import matplotlib.pyplot as plt
from local_utils import detect_lp
from os.path import splitext,basename
from keras.models import model_from_json
import glob
# !sudo apt install tesseract-ocr
# !pip install pytesseract

def load_model(path):
	path = splitext(path)[0]
	with open('%s.json' % path, 'r') as json_file:
		model_json = json_file.read()
	model = model_from_json(model_json, custom_objects={})
	model.load_weights('%s.h5' % path)
	print("Model load")
	return model

pretrained_obj_detector_net_path = "pretrained_obj_detector-net.json"
pretrained_obj_detector_net = load_model(pretrained_obj_detector_net_path)

def preprocess_image(image_path,resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img


def get_plate(image_path, Dmax=608, Dmin=256):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(pretrained_obj_detector_net, vehicle, bound_dim, lp_threshold=0.5)
    return LpImg, cor


def execute_object_detection(image_path):
    LpImg,_ = get_plate(image_path)
    plt.axis(False)
    gray = get_grayscale(LpImg[0])
    #plt.imshow(LpImg[0])
    plt.imsave('0001.jpg',gray)

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def get_car_number_plate(raw_img):
	execute_object_detection(raw_img)
	crop_img_loc = '0001.jpg'
	#text = pytesseract.image_to_string(crop_img_loc)
    custom_config = r'-l eng --psm 6'
    text = pytesseract.image_to_string(crop_img_loc, config=custom_config)

	return text
	# return 'RJ 14 8901'
