import os
import re

import numpy as np
from PIL import Image
# import tensorflow as tf
from tflite_micro_runtime.interpreter import Interpreter as tflm_Interpreter
# tflm_Interpreter = tf.lite.Interpreter

from quanteyes.dataloader.dataset_tf import get_zipped_dataset
from quanteyes.training_tf.utils import DATA_PATHS

tf_model_path = '../model_export/q-int8_d-2bit-octree.tflite'
interpreter = tflm_Interpreter(tf_model_path)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_tensor_index = input_details["index"]
output = interpreter.get_tensor(output_details["index"])

mse = 0.0
avg_cosine_similarity = 0.0
num_examples = 100
# it = iter(val_dataset.batch(1).take(num_examples))

img = Image.open('test.png')
input_data = np.array(img)
input_data = np.expand_dims(input_data, axis=0)
input_data = np.expand_dims(input_data, axis=-1)

img_array = input_data

input_scale, input_zero_point = input_details["quantization"]
if (input_scale, input_zero_point) != (0.0, 0):
	img_array = np.multiply(img_array, 1.0 / input_scale) + input_zero_point
img_array = img_array.astype(input_details["dtype"])

interpreter.set_tensor(input_tensor_index, img_array)
interpreter.invoke()
pred = output[0]

print(output, pred)

# If required, dequantized the output layer (from integer to float)
output_scale, output_zero_point = output_details["quantization"]
if (output_scale, output_zero_point) != (0.0, 0):
	pred = pred.astype(np.float32)
	pred = np.multiply((pred - output_zero_point), output_scale)

# Compute cosine similarity.
y_pred = pred.astype(np.float32)

print(y_pred)