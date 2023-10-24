import torch
import torch.nn as nn

class CNNLSTMModel(nn.Module):
    def __init__(self, static_model_path, train_backbone=False, seq_len=50, output_len=5):
        super(CNNLSTMModel, self).__init__()

        # Loading our pre-trained static backbone model (assuming it's a PyTorch model)
        self.resnet_no_top = torch.load(static_model_path)
        
        # Remove the dense layers from the model
        self.resnet_no_top = nn.Sequential(*list(self.resnet_no_top.children())[:-1])

        if not train_backbone:
            for param in self.resnet_no_top.parameters():
                param.requires_grad = False

        self.seq_len = seq_len
        self.lstm_enc = nn.LSTM(64, 128)
        self.lstm_dec = nn.LSTM(seq_len * 128, 128)
        self.output_len = output_len
 
        self.fc = nn.Linear(128, 3)

    def forward(self, x):
        # Input shape: (batch_size, seq_len, 400, 640, 1)
        batch_size, seq_len, _, _, _ = x.size()
        
        # Reshape the input
        x = x.view(batch_size * seq_len, 3, 400, 640)
        
        # Pass through the statifc backbonem and unshape.
        x = self.resnet_no_top(x)
        x = x.view(batch_size, seq_len, -1)
        
        # Pass through LSTM encoder 
        x, _ = self.lstm_enc(x)
        x = x.unsqueeze(1).repeat(1, self.output_len, 1, 1)
        x = x.view(batch_size, self.output_len, seq_len * 128)
        
        x, _ = self.lstm_dec(x)
        x = self.fc(x)

        return x
