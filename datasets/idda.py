import os
from typing import Any
import numpy as np
from PIL import Image
from torch import from_numpy
from torchvision.datasets import VisionDataset
import datasets.ss_transforms as tr

class_eval = [
    255,
    2,
    4,
    255,
    11,
    5,
    0,
    0,
    1,
    8,
    13,
    3,
    7,
    6,
    255,
    255,
    15,
    14,
    12,
    9,
    10,
]


class IDDADataset(VisionDataset):
    def __init__(
        self,
        root: str,
        list_samples: list[str],
        transform: tr.Compose = None,
        client_name: str = None,
    ):
        super().__init__(root=root, transform=transform, target_transform=None)
        self.list_samples = list_samples
        self.client_name = client_name
        self.target_transform = self.get_mapping()

    @staticmethod
    def get_mapping():
        classes = class_eval
        mapping = np.zeros((256,), dtype=np.int64) + 255
        for i, cl in enumerate(classes):
            mapping[i] = cl
        return lambda x: from_numpy(mapping[x])

    def __getitem__(self, index: int) -> Any:
        dataRoot = "/content/drive/MyDrive/MLDL_Datasets/idda/"  # Data root
        img_name = self.list_samples[index]
        img = Image.open(
            dataRoot + "images/" + img_name + ".jpg", "r"
        )  # Image at index 'index'
        target = Image.open(dataRoot + "labels/" + img_name + ".png", "r")
        img, target = self.transform(img, target)
        target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        return len(self.list_samples)
