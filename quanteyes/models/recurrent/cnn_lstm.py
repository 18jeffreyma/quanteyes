import torch
import torch.nn as nn

class CNNLSTMModel(nn.Module):
    def __init__(self, static_model_path, train_backbone=False, seq_len=50):
        super(CNNLSTMModel, self).__init__()

        # Loading our pre-trained static backbone model (assuming it's a PyTorch model)
        self.resnet_no_top = torch.load(static_model_path)
        
        # Remove the dense layers from the model
        self.resnet_no_top = nn.Sequential(*list(self.resnet_no_top.children())[:-1])

        if not train_backbone:
            for param in self.resnet_no_top.parameters():
                param.requires_grad = False

        self.seq_len = seq_len
        self.lstm_enc = nn.LSTM(2048, 128)
        self.repeater = nn.LSTM(128, 128)
        self.fc_1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_2 = nn.Linear(64, 3)

    def forward(self, x):
        print(x.shape)
        # Input shape: (batch_size, seq_len, 400, 640, 1)
        batch_size, seq_len, _, _, _ = x.size()
        
        # Reshape the input
        x = x.view(batch_size * seq_len, 3, 400, 640)
        
        # Pass through the static backbone
        x = self.resnet_no_top(x)
        x = x.view(batch_size, seq_len, -1)
        
        # TODO: jeff confused about this part
        
        # Pass through the LSTM encoder
        x, _ = self.lstm_enc(x)
        
        # Repeat the LSTM encoder output
        x = x.view(batch_size, 1, -1)
        x, _ = self.repeater(x)
        x = x.view(batch_size, seq_len, -1)
        
        # Pass through fully connected layers
        x = x.view(batch_size, seq_len, -1)
        x = self.fc_1(x)
        x = self.fc_2(x)
        
        # Reshape the output to (batch_size, seq_len, 3)
        x = x.view(batch_size, seq_len, 3)
        return x

