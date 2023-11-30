from picamera2 import Picamera2
import numpy as np
from tflite_runtime.interpreter import Interpreter
import time

class EyeTracker():
	def __init__(self, model_path=None):
		# camera related stuff
		self.picam2 			= None
		self.stride 			= None
		self.res 				= (160, 100)

		# tf stuff
		self.model_path 		= model_path if model_path is not None else '../model_export/q-int8_d-2bit-octree.tflite'
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

	def get_input(self):
		buffer = self.picam2.capture_buffer("lores")
		grey = buffer[:self.stride*self.res[1]].reshape((self.res[1], self.stride))[:, :self.res[0]]
		grey = np.expand_dims(grey, axis=0)
		return np.expand_dims(grey, axis=-1)

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

	def main(self):
		self.picam2.start()
		frames = 0
		
		self.start = time.time()
		while True:
			input_data = self.get_input()
			prediction = self.make_prediction(input_data)
			# print(prediction)

			frames += 1
			if frames % self.frames_per_print == 0:
				self.print_stats(frames)
				frames = 0

		self.picam2.stop()


def main():
	eyetracker = EyeTracker()
	eyetracker.setup()
	eyetracker.main()

if __name__ == '__main__':
	main()