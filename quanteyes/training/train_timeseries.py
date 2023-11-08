import os
import torch
import torch.nn as nn
import torch.optim as optim

from quanteyes.dataloader.dataset import OpenEDSDataset
from quanteyes.models import ResNet18

import logging

from quanteyes.models.recurrent.cnn_lstm_v2 import CNNLSTMModelV2
logging.basicConfig(level=logging.DEBUG)

# Define your training loop
def train(model, optimizer, criterion, train_loader, test_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print(f"iteration: {batch_idx}, loss: {loss.item()}")
        
        if (batch_idx + 1) % 250 == 0:
            model.eval()
            with torch.no_grad():
                total_count = 0
                total_loss = 0.
                for data, target in test_loader:
                    if (total_count == 20):
                        break
                    output = model(data)
                    total_loss += criterion(output, target)
                    total_count += 1
                print(f"validation loss: {total_loss/total_count}")

            torch.save(model.state_dict(), f"quanteyes/training/saved/cnn_lstm_v2_{batch_idx + 1}.pth")
            model.train()

# Define your data loading and preprocessing
train_path = "/mnt/sdb/data/Openedsdata2020/openEDS2020-GazePrediction/train"
train_dataset = OpenEDSDataset(
    os.path.join(train_path, "sequences"), 
    os.path.join(train_path, "labels"),
    inference=False,
    input_output_lengths=(5, 2),
    device="cuda")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

val_path = "/mnt/sdb/data/Openedsdata2020/openEDS2020-GazePrediction/validation"
val_dataset = OpenEDSDataset(
    os.path.join(train_path, "sequences"), 
    os.path.join(train_path, "labels"),
    inference=False,
    input_output_lengths=(5, 2),
    device="cuda")
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)

# Define your optimizer and loss function
model = CNNLSTMModelV2("quanteyes/training/saved/SimpleCNN_250.pth", train_backbone=False, seq_len=5, output_len=2).to("cuda")
optimizer = optim.Adam(model.parameters(), lr=0.001)

criterion = nn.MSELoss()

# Train your model
for epoch in range(10):
    train(model, optimizer, criterion, train_loader, val_loader)
