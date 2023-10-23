import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
from torchvision import io

class CustomDataset(Dataset):
    def __init__(self, data_dir, label_dir=None, seq_input_length=50, inference=False):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.seq_input_length = seq_input_length
        self.inference = inference
        self.data_paths, self.labels = self.generate_ids(self.data_dir, self.label_dir, bin_by_sequence=True, inference=inference)

    def generate_ids(self, data_files, label_files, bin_by_sequence=True, inference=False):
        data_list = []
        labels_list = []
        sequence_ids = sorted(os.listdir(data_files))
        label_ids = [x + ".txt" for x in sequence_ids]

        for sequence, label in zip(sequence_ids, label_ids):
            samples = sorted(os.listdir(os.path.join(data_files, sequence)))
            if not inference:
                labels = pd.read_csv(os.path.join(label_files, label), delimiter=',', skiprows=0, names=('id', 'x', 'y', 'z'), dtype={'id': str})
                labels.set_index('id', inplace=True)
                targets_list = []

            images = []

            for sample in samples:
                sample_name = os.path.splitext(sample)[0]
                sample_id = sequence + "_" + sample_name
                images.append(os.path.join(data_files, sequence, sample))

                if not inference:
                    targets_list.append(labels.loc[[sample_name]].values[0])

            data_list.append(images)

            if not inference:
                labels_list.append(targets_list)

        if not bin_by_sequence:
            flat_data_list = [item for sublist in data_list for item in sublist]
            flat_labels_list = [item for sublist in labels_list for item in sublist]
            return flat_data_list, flat_labels_list

        return data_list, labels_list

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data = self.data_paths[idx]
        images = []

        for path in data:
            img = io.read_image(path)
            img = img.float() / 255.0  # Normalize to [0, 1]
            img = img.unsqueeze(0)  # Add a channel dimension
            images.append(img)

        if self.inference:
            return images

        labels = self.labels[idx]
        return images, labels

def get_dataset(data_dir, batch_size=16, seq_input_length=50, inference=False):
    dataset = CustomDataset(data_dir, seq_input_length=seq_input_length, inference=inference)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=not inference)

    return dataloader, len(dataset)

if __name__ == "__main__":
    