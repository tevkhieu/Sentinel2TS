import os
import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset
import numpy as np


class TimeSeriesDataset(Dataset):
    """Dataset class for training an autoencoder on Sentinel 2 data"""

    def __init__(
        self,
        dataset_path: str,
        minimal_x: int = 250,
        maximal_x: int = 400,
        minimal_y: int = 250,
        maximal_y: int = 400,
    ) -> None:
        """
        Initialize the dataset class

        Args:
            path (str): path to the data
        Returns:
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.minimal_x = minimal_x
        self.maximal_x = maximal_x
        self.minimal_y = minimal_y
        self.maximal_y = maximal_y

        self.dataset_len = (maximal_x - minimal_x) * (maximal_y - minimal_y)

    def __len__(self) -> int:
        return self.dataset_len

    def __getitem__(self, index: int) -> Tensor:
        initial_x = (index // (self.maximal_x - self.minimal_x)) + self.minimal_x
        initial_y = (index % (self.maximal_x - self.minimal_x)) + self.minimal_y
        data_path = os.path.join(
            self.dataset_path, f"{initial_x:03}_{initial_y:03}.npy"
        )
        data = np.load(data_path)

        return torch.Tensor(data).swapaxes(0, 1)[:, 1:], initial_x, initial_y
