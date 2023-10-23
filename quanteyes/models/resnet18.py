import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self, fc1_units, drop_out):
        super(ResNet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=False)
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Identity()  # Remove the final fully connected layer

        self.fc1 = nn.Linear(num_ftrs, fc1_units)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_out)
        self.out = nn.Linear(fc1_units, 3)

    def forward(self, x):
        x = self.resnet18(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.out(x)
        return out
