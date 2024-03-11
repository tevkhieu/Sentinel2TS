import os
import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset
import numpy as np

from sentinel2_ts.utils.process_data import get_state, get_state_time_series


class SentinelDataset(Dataset):
    """Dataset class"""

    def __init__(
        self,
        dataset_path: str,
        data_sequence_length: int = 341,
        time_prediction_length: int = 100,
        dataset_len: int = 512 * 512,
        minimal_x: int = 250,
        maximal_x: int = 400,
        minimal_y: int = 250,
        maximal_y: int = 400,
    ) -> None:
        """
        Initialize the dataset class

        Args:
            path (str): path to the data
            clipping(bool): whether the values above 1 should be clipped to 1 or not
        Returns:
            None
        """
        super().__init__()
        self.dataset_path = dataset_path

        self.time_prediction_length = time_prediction_length
        self.dataset_len = dataset_len

        self.initial_times = torch.randint(
            low=1, high=data_sequence_length - 2 * self.time_prediction_length, size=(dataset_len,)
        )
        self.initial_x = torch.randint(low=minimal_x, high=maximal_x, size=(dataset_len,))
        self.initial_y = torch.randint(low=minimal_y, high=maximal_y, size=(dataset_len,))

    def __len__(self) -> int:
        return self.dataset_len

    def __getitem__(self, index: int) -> Tensor:
        initial_time = self.initial_times[index]
        initial_x = self.initial_x[index]
        initial_y = self.initial_y[index]
        data_path = os.path.join(self.dataset_path, f"{initial_x}_{initial_y}.npy")
        data = np.load(data_path)
        initial_state = get_state(data, initial_time)

        observed_state_time_series = get_state_time_series(data, initial_time, self.time_prediction_length)
        return initial_state.unsqueeze(0), observed_state_time_series
