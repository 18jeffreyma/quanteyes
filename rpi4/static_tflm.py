import os

import numpy as np
from PIL import Image
# import tensorflow as tf
import time
from tflite_runtime.interpreter import Interpreter as tflm_Interpreter
# tflm_Interpreter = tf.lite.Interpreter

def main():
	tf_model_path = '../model_export/q-int8_d-2bit-octree.tflite'
	interpreter = tflm_Interpreter(tf_model_path)
	interpreter.allocate_tensors()

	data_path = '../data/2bit-octree/'

	# Get input and output tensors
	input_details = interpreter.get_input_details()[0]
	output_details = interpreter.get_output_details()[0]
	input_scale, input_zero_point = input_details["quantization"]

	input_tensor_index = input_details["index"]
	output = interpreter.get_tensor(output_details["index"])

	# downsample
	images = []
	for file in os.listdir(data_path):
		input_data = Image.open(os.path.join(data_path, file))
		input_data = np.array(input_data)
		input_data = np.expand_dims(input_data, axis=0)
		input_data = np.expand_dims(input_data, axis=-1)

		if (input_scale, input_zero_point) != (0.0, 0):
			input_data = np.multiply(input_data, 1.0 / input_scale) + input_zero_point
		input_data = input_data.astype(input_details["dtype"])
		images.append(input_data)

	# do inference
	start = time.time()
	for img_array in images:

		interpreter.set_tensor(input_tensor_index, img_array)
		interpreter.invoke()
		output = interpreter.get_tensor(output_details["index"])
		pred = output[0]

		# If required, dequantized the output layer (from integer to float)
		output_scale, output_zero_point = output_details["quantization"]
		if (output_scale, output_zero_point) != (0.0, 0):
			pred = pred.astype(np.float32)
			pred = np.multiply((pred - output_zero_point), output_scale)

		# Compute cosine similarity.
		y_pred = pred.astype(np.float32)
		print(y_pred)
	end = time.time()
	elapsed = end - start
	print('elapsed time: ', elapsed)
	print('time per image: ', elapsed/len(images))
	print('hertz: ', len(images)/elapsed)

if __name__ == '__main__':
	main()