import os
from typing import Any
import numpy as np
from PIL import Image
from torch import from_numpy
from torchvision.datasets import VisionDataset
import datasets.ss_transforms as tr

class_map = {
   1: 13,  # ego_vehicle : vehicle
   7: 0,   # road
   8: 1,   # sidewalk
   11: 2,  # building
   12: 3,  # wall
   13: 4,  # fence
   17: 5,  # pole
   18: 5,  # poleGroup: pole
   19: 6,  # traffic light
   20: 7,  # traffic sign
   21: 8,  # vegetation
   22: 9,  # terrain
   23: 10,  # sky
   24: 11,  # person
   25: 12,  # rider
   26: 13,  # car: vehicle
   27: 13,  # truck: vehicle
   28: 13,  # bus: vehicle
   32: 14,  # motorcycle
   33: 15,  # bicycle
}


class GTADataset(VisionDataset):
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
        classes = class_map
        mapping = np.zeros((256,), dtype=np.int64) + 255
        for i, cl in classes.items():
            mapping[i] = cl
        return lambda x: from_numpy(mapping[x])

    def __getitem__(self, index: int) -> Any:
        dataRoot = "/content/drive/MyDrive/MLDL_Datasets/GTA5/"  # Data root
        img_name = self.list_samples[index]
        img = Image.open(
            dataRoot + "images/" + img_name, "r"
        )  # Image at index 'index'
        target = Image.open(dataRoot + "labels/" + img_name, "r")
        img, target = self.transform(img, target)
        target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        return len(self.list_samples)
