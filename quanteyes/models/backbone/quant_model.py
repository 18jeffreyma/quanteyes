import copy
import math
import random
from collections import OrderedDict, defaultdict

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchprofile import profile_macs
from torchvision.datasets import *
from torchvision.transforms import *

from torchprofile import profile_macs

assert torch.cuda.is_available(), \
"The current runtime does not have CUDA support." \
"Please go to menu bar (Runtime - Change runtime type) and select GPU"

def get_quantized_range(bitwidth):
	quantized_max = (1 << (bitwidth - 1)) - 1
	quantized_min = -(1 << (bitwidth - 1))
	return quantized_min, quantized_max

class QuantizedConv2d(nn.Module):
	def __init__(self, weight, bias,
				 input_zero_point, output_zero_point,
				 input_scale, weight_scale, output_scale,
				 stride, padding, dilation, groups,
				 feature_bitwidth=8, weight_bitwidth=8):
		super().__init__()
		# current version Pytorch does not support IntTensor as nn.Parameter
		self.register_buffer('weight', weight)
		self.register_buffer('bias', bias)

		self.input_zero_point = input_zero_point
		self.output_zero_point = output_zero_point

		self.input_scale = input_scale
		self.register_buffer('weight_scale', weight_scale)
		self.output_scale = output_scale

		self.stride = stride
		self.padding = (padding[1], padding[1], padding[0], padding[0])
		self.dilation = dilation
		self.groups = groups

		self.feature_bitwidth = feature_bitwidth
		self.weight_bitwidth = weight_bitwidth

	def forward(self, x):
		return quantized_conv2d(
			x, self.weight, self.bias,
			self.feature_bitwidth, self.weight_bitwidth,
			self.input_zero_point, self.output_zero_point,
			self.input_scale, self.weight_scale, self.output_scale,
			self.stride, self.padding, self.dilation, self.groups
			)

	def quantized_conv2d(input, weight, bias, feature_bitwidth, weight_bitwidth,
						input_zero_point, output_zero_point,
						input_scale, weight_scale, output_scale,
						stride, padding, dilation, groups):
		"""
		quantized 2d convolution
		:param input: [torch.CharTensor] quantized input (torch.int8)
		:param weight: [torch.CharTensor] quantized weight (torch.int8)
		:param bias: [torch.IntTensor] shifted quantized bias or None (torch.int32)
		:param feature_bitwidth: [int] quantization bit width of input and output
		:param weight_bitwidth: [int] quantization bit width of weight
		:param input_zero_point: [int] input zero point
		:param output_zero_point: [int] output zero point
		:param input_scale: [float] input feature scale
		:param weight_scale: [torch.FloatTensor] weight per-channel scale
		:param output_scale: [float] output feature scale
		:return:
			[torch.(cuda.)CharTensor] quantized output feature
		"""
		assert(len(padding) == 4)
		assert(input.dtype == torch.int8)
		assert(weight.dtype == input.dtype)
		assert(bias is None or bias.dtype == torch.int32)
		assert(isinstance(input_zero_point, int))
		assert(isinstance(output_zero_point, int))
		assert(isinstance(input_scale, float))
		assert(isinstance(output_scale, float))
		assert(weight_scale.dtype == torch.float)

		# Step 1: calculate integer-based 2d convolution (8-bit multiplication with 32-bit accumulation)
		input = torch.nn.functional.pad(input, padding, 'constant', input_zero_point)
		if 'cpu' in input.device.type:
			# use 32-b MAC for simplicity
			output = torch.nn.functional.conv2d(input.to(torch.int32), weight.to(torch.int32), None, stride, 0, dilation, groups)
		else:
			# current version pytorch does not yet support integer-based conv2d() on GPUs
			output = torch.nn.functional.conv2d(input.float(), weight.float(), None, stride, 0, dilation, groups)
			output = output.round().to(torch.int32)
		if bias is not None:
			output = output + bias.view(1, -1, 1, 1)

		############### YOUR CODE STARTS HERE ###############
		# hint: this code block should be the very similar to quantized_linear()

		# Step 2: scale the output
		#         hint: 1. scales are floating numbers, we need to convert output to float as well
		#               2. the shape of weight scale is [oc, 1, 1, 1] while the shape of output is [batch_size, oc, height, width]
		output = output.float() * (input_scale * weight_scale / output_scale).view(1, -1, 1, 1)

		# Step 3: shift output by output_zero_point
		#         hint: one line of code
		output = output + output_zero_point
		############### YOUR CODE ENDS HERE #################

		# Make sure all value lies in the bitwidth-bit range
		output = output.round().clamp(*get_quantized_range(feature_bitwidth)).to(torch.int8)
		return output

class QuantizedLinear(nn.Module):
	def __init__(self, weight, bias,
				 input_zero_point, output_zero_point,
				 input_scale, weight_scale, output_scale,
				 feature_bitwidth=8, weight_bitwidth=8):
		super().__init__()
		# current version Pytorch does not support IntTensor as nn.Parameter
		self.register_buffer('weight', weight)
		self.register_buffer('bias', bias)

		self.input_zero_point = input_zero_point
		self.output_zero_point = output_zero_point

		self.input_scale = input_scale
		self.register_buffer('weight_scale', weight_scale)
		self.output_scale = output_scale

		self.feature_bitwidth = feature_bitwidth
		self.weight_bitwidth = weight_bitwidth

		

	def forward(self, x):
		return quantized_linear(
			x, self.weight, self.bias,
			self.feature_bitwidth, self.weight_bitwidth,
			self.input_zero_point, self.output_zero_point,
			self.input_scale, self.weight_scale, self.output_scale
			)
	
	def quantized_linear(input, weight, bias, feature_bitwidth, weight_bitwidth,
					 input_zero_point, output_zero_point,
					 input_scale, weight_scale, output_scale):
		"""
		quantized fully-connected layer
		:param input: [torch.CharTensor] quantized input (torch.int8)
		:param weight: [torch.CharTensor] quantized weight (torch.int8)
		:param bias: [torch.IntTensor] shifted quantized bias or None (torch.int32)
		:param feature_bitwidth: [int] quantization bit width of input and output
		:param weight_bitwidth: [int] quantization bit width of weight
		:param input_zero_point: [int] input zero point
		:param output_zero_point: [int] output zero point
		:param input_scale: [float] input feature scale
		:param weight_scale: [torch.FloatTensor] weight per-channel scale
		:param output_scale: [float] output feature scale
		:return:
			[torch.CharIntTensor] quantized output feature (torch.int8)
		"""
		assert(input.dtype == torch.int8)
		assert(weight.dtype == input.dtype)
		assert(bias is None or bias.dtype == torch.int32)
		assert(isinstance(input_zero_point, int))
		assert(isinstance(output_zero_point, int))
		assert(isinstance(input_scale, float))
		assert(isinstance(output_scale, float))
		assert(weight_scale.dtype == torch.float)

		# Step 1: integer-based fully-connected (8-bit multiplication with 32-bit accumulation)
		if 'cpu' in input.device.type:
			# use 32-b MAC for simplicity
			output = torch.nn.functional.linear(input.to(torch.int32), weight.to(torch.int32), bias)
		else:
			# current version pytorch does not yet support integer-based linear() on GPUs
			output = torch.nn.functional.linear(input.float(), weight.float(), bias.float())

		############### YOUR CODE STARTS HERE ###############
		# Step 2: scale the output
		#         hint: 1. scales are floating numbers, we need to convert output to float as well
		#               2. the shape of weight scale is [oc, 1, 1, 1] while the shape of output is [batch_size, oc]
		output = output.float() * (input_scale * weight_scale / output_scale).view(1, -1)

		# Step 3: shift output by output_zero_point
		#         hint: one line of code
		output = output + output_zero_point
		############### YOUR CODE ENDS HERE #################

		# Make sure all value lies in the bitwidth-bit range
		output = output.round().clamp(*get_quantized_range(feature_bitwidth)).to(torch.int8)
		return output

class QuantizedMaxPool2d(nn.MaxPool2d):
	def forward(self, x):
		# current version PyTorch does not support integer-based MaxPool
		return super().forward(x.float()).to(torch.int8)

class QuantizedAvgPool2d(nn.AvgPool2d):
	def forward(self, x):
		# current version PyTorch does not support integer-based AvgPool
		return super().forward(x.float()).to(torch.int8)
