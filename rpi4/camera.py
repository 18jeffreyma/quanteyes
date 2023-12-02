from picamera2 import MappedArray, Picamera2, Preview
import numpy as np
from tflite_runtime.interpreter import Interpreter
import time
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import threading
from collections import Counter

class EyeTracker():
	def __init__(self, model_path=None, data_path=None, preview=True):
		# camera related stuff
		self.picam2 			= None
		self.stride 			= None
		self.preview			= preview
		self.text				= []
		self.prev_text			= []
		self.res 				= (160, 100)

		# tf stuff
		self.model_path 		= model_path if model_path is not None else '../model_export/q-int8_d-2bit-octree.tflite'
		self.data_path			= data_path if data_path is not None else '../data/2bit-octree'
		self.interpreter 		= None
		self.input_details 		= None
		self.output_details 	= None
		self.input_scale 		= None
		self.input_zero_point 	= None
		self.input_tensor_index = None
		self.output_scale		= None
		self.output_zero_point	= None

		# misc
		self.start				= None
		self.frames_per_print	= 5

	def setup(self):
		self.setup_picam()
		self.setup_tf()


	def setup_picam(self):
		self.picam2 	= Picamera2()

		if self.preview:
			os.environ['DISPLAY'] = ':0'
			self.picam2.start_preview(Preview.QTGL)
			self.picam2.post_callback = self.add_text_to_frame
		preview_config 	= self.picam2.create_preview_configuration(
			main={"size": self.res}, 
			lores={"size": self.res, "format": "YUV420"}
		)
		self.picam2.configure(preview_config)
		self.stride 	= self.picam2.stream_configuration("lores")["stride"]

	def setup_tf(self):
		self.interpreter = Interpreter(self.model_path)
		self.interpreter.allocate_tensors()
		self.input_details 	= self.interpreter.get_input_details()[0]
		self.output_details = self.interpreter.get_output_details()[0]
		self.input_scale, self.input_zero_point = self.input_details["quantization"]
		self.input_tensor_index = self.input_details["index"]
		self.output_scale, self.output_zero_point = self.output_details["quantization"]

	def get_camera_input(self):
		buffer = self.picam2.capture_buffer("lores")
		grey = buffer[:self.stride*self.res[1]].reshape((self.res[1], self.stride))[:, :self.res[0]]
		return self.expand_dims(grey)

	def expand_dims(self, arr):
		arr = np.expand_dims(arr, axis=0)
		return np.expand_dims(arr, axis=-1)

	def make_prediction(self, input_data):
		if (self.input_scale, self.input_zero_point) != (0.0, 0):
			input_data = np.multiply(input_data, 1.0 / self.input_scale) + self.input_zero_point
		input_data = input_data.astype(self.input_details["dtype"])
		self.interpreter.set_tensor(0, input_data)
		self.interpreter.invoke()
		output = self.interpreter.get_tensor(self.output_details["index"])
		if (self.output_scale, self.output_zero_point) != (0.0, 0):
			output = output.astype(np.float32)
			output = np.multiply((output - self.output_zero_point), self.output_scale)
		return output[0]

	def print_stats(self, frames):
		end = time.time()
		elapsed = end - self.start
		# print(f'elapsed time: {elapsed}', end='\r')
		# print(f'time per image: {elapsed/frames}', end='\r')
		print(f'hertz: {(frames/elapsed):.2f}', end='\r')

		self.start = time.time()

	def add_text_to_frame(self, request):
		with MappedArray(request, "main") as m:
			# print(np.array(m.array).shape)

			y = 10
			text_arr = self.text if self.text != [] else self.prev_text
			for text in text_arr:
				cv2.putText(m.array, text, (0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1, cv2.LINE_AA)
				y += 20
			
			if self.text != []:
				self.prev_text = self.text
				self.text = []

	def one_by_one(self):
		self.picam2.start()
		moved = False
		# plt.figure(figsize=(20, 10))
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

		for file in os.listdir(self.data_path):
			input_data = Image.open(os.path.join(self.data_path, file))
			# mngr = plt.get_current_fig_manager()
			# mngr.window.setGeometry(50,100,640, 545)
			ax1.imshow(input_data, cmap='gray')
			plt.pause(0.0001)

			if not moved:
				input("Press enter when done:")
				moved = True

			# do static
			prediction_static = self.make_prediction(self.expand_dims(input_data))

			# do camera
			camera_input = self.get_camera_input()
			camera_input = np.where(camera_input < 50, 0, camera_input)
			print(Counter(camera_input.flatten())[0])
			prediction_camera = self.make_prediction(camera_input)

			ax2.imshow(camera_input[0, :, :, 0], cmap='gray')
			plt.pause(0.0001)


			self.text.append(f'[{prediction_static[0]:.2f}, {prediction_static[1]:.2f}, {prediction_static[2]:.2f}]')
			self.text.append(f'[{prediction_camera[0]:.2f}, {prediction_camera[1]:.2f}, {prediction_camera[2]:.2f}]')

		self.picam2.stop()

	def realtime_loop(self):
		self.picam2.start()
		frames = 0
		
		self.start = time.time()
		while True:
			input_data = self.get_camera_input()
			prediction = self.make_prediction(input_data)
			self.text.append(f'[{prediction[0]:.2f}, {prediction[1]:.2f}, {prediction[2]:.2f}]')
			# print(prediction)

			frames += 1
			if frames % self.frames_per_print == 0:
				self.print_stats(frames)
				frames = 0

		self.picam2.stop()


def main():
	eyetracker = EyeTracker()
	eyetracker.setup()
	eyetracker.one_by_one()
	# eyetracker.realtime_loop()

if __name__ == '__main__':
	main()