from typing import List

import numpy as np
from torch.utils.data import Dataset

from src.data.preprocessing.base_preprocessing import load_yaml


class THREED_FRONTPC(Dataset):
    def __init__(self, data_path, train: bool, transform=None):
        mode = "train" if train else "test"

        data_attr = load_yaml(f"{data_path}/{mode}_database.yaml")
        self.data_path_input: List[str] = [
            attr["input"] for attr in data_attr
        ]
        self.data_path_gt: List[str] = [
            attr["gt"] for attr in data_attr
        ]

        self.transform = transform

    def __len__(self):
        return len(self.data_path_input)

    def __getitem__(self, index):
        in_xyz = np.load(self.data_path_input[index])
        gt_xyzl = np.load(self.data_path_gt[index])

        return in_xyz, gt_xyzl

class _THREED_FRONTPC(Dataset):
    def __init__(self, data_path, train: bool, transform=None):
        mode = "train" if train else "test"

        data_attr = load_yaml(f"{data_path}/camera_move_test_database.yaml")
        self.data_path_input: List[str] = [
            attr["input"] for attr in data_attr
        ]
        self.data_path_gt: List[str] = [
            attr["gt"] for attr in data_attr
        ]

        self.transform = transform

    def __len__(self):
        return len(self.data_path_input)

    def __getitem__(self, index):
        in_xyz = np.load(self.data_path_input[index])
        gt_xyzl = np.load(self.data_path_gt[index])

        return in_xyz, gt_xyzl
