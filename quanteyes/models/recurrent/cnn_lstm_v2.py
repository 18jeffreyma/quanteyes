import torch
import torch.nn as nn
from brevitas.nn import QuantIdentity, QuantLinear, QuantLSTM
from torchsummary import summary


class CNNLSTMModelV2(nn.Module):
    def __init__(
        self, static_model_path, train_backbone=False, seq_len=50, output_len=5
    ):
        super(CNNLSTMModelV2, self).__init__()

        # Loading our pre-trained static backbone model (assuming it's a PyTorch model)
        self.cnn_no_final_layer = torch.load(static_model_path)

        self.cnn_no_final_layer.fc3 = nn.Identity(0)

        # Print the summary of the modified model
        summary(self.cnn_no_final_layer, (3, 400, 640))

        if not train_backbone:
            for param in self.cnn_no_final_layer.parameters():
                param.requires_grad = False

        self.seq_len = seq_len
        self.lstm_enc = nn.LSTM(128, 128)
        self.lstm_dec = nn.LSTM(seq_len * 128, 128)
        self.output_len = output_len

        self.fc = nn.Linear(128, 3)

    def forward(self, x):
        # Input shape: (batch_size, seq_len, 400, 640, 1)
        batch_size, seq_len, _, _, _ = x.size()

        # Reshape the input
        x = x.view(batch_size * seq_len, 3, 400, 640)

        # Pass through the statifc backbonem and unshape.
        x = self.cnn_no_final_layer(x)

        x = x.view(batch_size, seq_len, -1)

        # Pass through LSTM encoder
        x, _ = self.lstm_enc(x)
        x = x.unsqueeze(1).repeat(1, self.output_len, 1, 1)
        x = x.view(batch_size, self.output_len, seq_len * 128)

        x, _ = self.lstm_dec(x)
        x = self.fc(x)

        return x


class QuantizedCNNLSTMModelV2(nn.Module):
    def __init__(
        self, static_model_path, train_backbone=False, seq_len=50, output_len=5
    ):
        super(QuantizedCNNLSTMModelV2, self).__init__()

        bit_width = 4

        # Loading our pre-trained static backbone model (assuming it's a PyTorch model)
        self.cnn_no_final_layer = torch.load(static_model_path)

        self.cnn_no_final_layer.fc3 = QuantIdentity(
            weight_bit_width=bit_width, return_quant_tensor=True
        )

        if not train_backbone:
            for param in self.cnn_no_final_layer.parameters():
                param.requires_grad = False

        self.seq_len = seq_len
        self.lstm_enc = QuantLSTM(
            128, 128, weight_bit_width=bit_width, return_quant_tensor=True
        )
        self.lstm_dec = QuantLSTM(
            seq_len * 128, 128, weight_bit_width=bit_width, return_quant_tensor=True
        )
        self.output_len = output_len

        self.fc = QuantLinear(128, 3)

    def forward(self, x):
        # Input shape: (batch_size, seq_len, 400, 640, 1)
        batch_size, seq_len, _, _, _ = x.size()

        # Reshape the input
        x = x.view(batch_size * seq_len, 3, 400, 640)

        # Pass through the statifc backbonem and unshape.
        x = self.cnn_no_final_layer(x)

        x = x.view(batch_size, seq_len, -1)

        # Pass through LSTM encoder
        x, _ = self.lstm_enc(x)
        x = x.unsqueeze(1).repeat(1, self.output_len, 1, 1)
        x = x.view(batch_size, self.output_len, seq_len * 128)

        x, _ = self.lstm_dec(x)
        x = self.fc(x)

        return x
