import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim

from quanteyes.dataloader.dataset import OpenEDSDataset
from quanteyes.models.backbone.simple_cnn import SimpleQuantizedCNN

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
                total_loss = 0.0
                for data, target in test_loader:
                    if total_count == 20:
                        break
                    output = model(data)
                    total_loss += criterion(output, target)
                    total_count += 1
                print(f"validation loss: {total_loss/total_count}")

            torch.save(
                model,
                f"quanteyes/training/saved/{model.__class__.__name__}_{batch_idx + 1}.pth",
            )
            model.train()


# Define your data loading and preprocessing
base_path = "/data/openEDS2020-GazePrediction-2bit"
train_path = f"{base_path}/train"
train_dataset = OpenEDSDataset(
    os.path.join(train_path, "sequences"),
    os.path.join(train_path, "labels"),
    inference=False,
    device="cuda",
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

val_path = f"{base_path}/validation"
val_dataset = OpenEDSDataset(
    os.path.join(train_path, "sequences"),
    os.path.join(train_path, "labels"),
    inference=False,
    device="cuda",
)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True)

# Define your optimizer and loss function
model = SimpleQuantizedCNN().to("cuda")
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Train your model
for epoch in range(10):
    train(model, optimizer, criterion, train_loader, val_loader)
