import io
import os
from PIL import Image
from flask import Flask, render_template, request, send_file
import detector

app = Flask(__name__)

def run_inference(img_path = 'file.jpg'):
	result = detector.get_car_number_plate(img_path)
	result = "UP 14"
	return result

@app.route("/")
def index():
	return render_template('index.html')

	
def run_detection(request):
	result = run_inference('file.jpg')
	result = Image.open('0001.jpg')
	file_object = io.BytesIO()
	result.save(file_object, 'PNG')
	file_object.seek(0)
	return send_file(file_object, mimetype='image/jpeg')

	
def run_ocr(request):
	result = run_inference('file.jpg')
	return 'Car Number Plate is: {}'.format(result)
	


@app.route("/detect", methods=['POST'])
def upload():
	file = Image.open(request.files['file'].stream)
	rgb_im = file.convert('RGB')
	rgb_im.save('file.jpg')
	task = dict(request.form)['action']
	if task=="detect":
		return run_detection(request)
	if task=="ocr":
		return run_ocr(request)

	
if __name__ == "__main__":
	app.run()