import torch.nn as nn
from brevitas.nn import QuantConv2d, QuantLinear, QuantMaxPool2d, QuantReLU
from torch import flatten


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 100 * 160, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleQuantizedCNN(nn.Module):
    def __init__(self):
        super(SimpleQuantizedCNN, self).__init__()

        bit_width = 4
        self.conv1 = QuantConv2d(
            3,
            32,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            weight_bit_width=bit_width,
            return_quant_tensor=True,
        )

        self.pool = QuantMaxPool2d(kernel_size=2, stride=2, return_quant_tensor=True)
        self.conv2 = QuantConv2d(
            32,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            weight_bit_width=bit_width,
            return_quant_tensor=True,
        )
        self.fc1 = QuantLinear(
            64 * 100 * 160,
            128,
            True,
            weight_bit_width=bit_width,
            return_quant_tensor=True,
        )
        self.fc2 = QuantLinear(
            128, 128, True, weight_bit_width=bit_width, return_quant_tensor=True
        )
        self.fc3 = QuantLinear(
            128, 3, True, weight_bit_width=bit_width, return_quant_tensor=True
        )

        self.relu1 = QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.relu2 = QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.relu3 = QuantReLU(bit_width=bit_width, return_quant_tensor=True)
        self.relu4 = QuantReLU(bit_width=bit_width, return_quant_tensor=True)

    def forward(self, x):
        x = self.pool(self.relu1(self.conv1(x)))
        x = self.pool(self.relu2(self.conv2(x)))
        x = flatten(x, 1)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x


# from brevitas.export import export_onnx_qcdq
# import torch

# # Weight-only model
# export_onnx_qcdq(quant_weight_lenet, torch.randn(1, 3, 32, 32), export_path='4b_weight_lenet.onnx')

# # Weight-activation model
# export_onnx_qcdq(quant_weight_act_lenet, torch.randn(1, 3, 32, 32), export_path='4b_weight_act_lenet.onnx')

# # Weight-activation-bias model
# export_onnx_qcdq(quant_weight_act_bias_lenet, torch.randn(1, 3, 32, 32), export_path='4b_weight_act_bias_lenet.onnx')
