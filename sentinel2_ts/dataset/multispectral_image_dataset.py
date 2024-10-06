import os
import numpy as np
import torch
from torch.utils.data import Dataset


class MultispectralImageDataset(Dataset):
    def __init__(self, dataset_path: str) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.len = len(os.listdir(dataset_path))

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index: int):
        return torch.Tensor(np.load(os.path.join(self.dataset_path, f"{index:03}.npy"))), index
