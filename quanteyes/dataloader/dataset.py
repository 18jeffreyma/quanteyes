import os
from functools import lru_cache
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import io


@lru_cache(maxsize=32)
def read_image_to_tensor(image_path: str, device: str = "cpu") -> torch.Tensor:
    # Read image to PyTorch tensor
    img = io.read_image(image_path).to(device)

    # Normalize to [0, 1]
    img = img.float() / 255.0

    # Convert to RGB if grayscale
    img = img.repeat(3, 1, 1)

    return img


class OpenEDSDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        label_dir: Optional[str] = None,
        input_output_lengths: Optional[tuple] = None,
        inference: bool = False,
        device: str = "cpu",
    ):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.inference = inference

        self.input_output_lengths = input_output_lengths
        self.device = device

        data_paths, labels = self._get_data_paths(
            self.data_dir, self.label_dir, inference=inference
        )

        self.image_paths = []
        self.labels = []

        # Map data paths to rolling windows of image paths and output vectors.
        if input_output_lengths is not None:
            input_length, output_length = input_output_lengths
            for i, data_path in enumerate(data_paths):
                if (input_length + output_length) <= len(data_path):
                    for window_start in range(
                        len(data_path) - input_length - output_length
                    ):
                        window_images = data_path[
                            window_start : window_start + input_length
                        ]
                        self.image_paths.append(window_images)

                        if label_dir is not None:
                            window_labels = labels[i][
                                window_start
                                + input_length : window_start
                                + input_length
                                + output_length
                            ]
                            self.labels.append(window_labels)
        else:
            # Otherwise, just use 1-to-1 mapping of image path to corresponding label.
            for i, data_path in enumerate(data_paths):
                for j, im in enumerate(data_path):
                    self.image_paths.append(im)
                    if label_dir is not None:
                        self.labels.append(labels[i][j])

    def _get_data_paths(
        self, data_files, label_files, inference=False
    ) -> Tuple[List[Union[str, List[str]]], List[Union[np.ndarray, List[np.ndarray]]]]:
        """
        Returns a list of image paths and list or labels, inner nested by sequence.
        """
        data_list = []
        labels_list = []
        sequence_ids = sorted(os.listdir(data_files))
        label_ids = [x + ".txt" for x in sequence_ids]

        for sequence, label in zip(sequence_ids, label_ids):
            samples = sorted(os.listdir(os.path.join(data_files, sequence)))
            if not inference:
                labels = pd.read_csv(
                    os.path.join(label_files, label),
                    delimiter=",",
                    skiprows=0,
                    names=("id", "x", "y", "z"),
                    dtype={"id": str},
                )
                labels.set_index("id", inplace=True)
                targets_list = []

            images = []

            for sample in samples:
                sample_name = os.path.splitext(sample)[0]
                images.append(os.path.join(data_files, sequence, sample))

                if not inference:
                    targets_list.append(labels.loc[[sample_name]].values[0])

            data_list.append(images)

            if not inference:
                labels_list.append(targets_list)

        return data_list, labels_list

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path_or_paths = self.image_paths[idx]

        data = None
        if isinstance(path_or_paths, str):
            data = read_image_to_tensor(path_or_paths, device=self.device)
        else:
            data = torch.stack(
                [
                    read_image_to_tensor(path, device=self.device)
                    for path in path_or_paths
                ],
                dim=0,
            )

        if self.inference:
            return data

        labels = torch.Tensor(self.labels[idx]).to(device=self.device)
        return data, labels


if __name__ == "__main__":
    path = "/mnt/sdb/data/Openedsdata2020/openEDS2020-GazePrediction/train"
    dataset = OpenEDSDataset(
        os.path.join(path, "sequences"),
        os.path.join(path, "labels"),
        inference=False,
        input_output_lengths=(50, 10),
    )

    image, label = dataset[0]
    print(image.shape, label.shape)
