import os
import torch
import torch.nn as nn
import torch.optim as optim

from quanteyes.dataloader.dataset import OpenEDSDataset
from quanteyes.models import ResNet18

import logging
logging.basicConfig(level=logging.DEBUG)

# Define your training loop
def train(model, optimizer, criterion, train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print(f"iteration: {batch_idx}, loss: {loss.item()}")

# Define your data loading and preprocessing
train_path = "/mnt/sdb/data/Openedsdata2020/openEDS2020-GazePrediction/train"
train_dataset = OpenEDSDataset(
    os.path.join(train_path, "sequences"), 
    os.path.join(train_path, "labels"),
    inference=False,
    device="cuda")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define your optimizer and loss function
model = ResNet18(64, 0.5).to("cuda")
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Train your model
for epoch in range(10):
    train(model, optimizer, criterion, train_loader)
