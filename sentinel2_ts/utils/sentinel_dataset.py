import os
import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset
import numpy as np

from sentinel2_ts.utils.process_data import scale_data, get_state

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
            maximal_y: int = 400
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
            low=1,
            high=data_sequence_length - 2 * self.time_prediction_length,
            size=(dataset_len,)
        )
        self.initial_x = torch.randint(low=minimal_x, high=maximal_x, size=(dataset_len,))
        self.initial_y = torch.randint(low=minimal_y, high=maximal_y, size=(dataset_len,))

    def __len__(self) -> int:
        return self.dataset_len
    
    def __getitem__(self, index: int) -> Tensor:
        intial_time = self.initial_times[index]
        initial_x = self.initial_x[index]
        initial_y = self.initial_y[index]
        data_path = os.path.join(self.dataset_path, f"{initial_x}_{initial_x}.npy")
        data = np.load(data_path)
        initial_state = get_state(data, initial_x, initial_y, intial_time)

        target_states = np.zeros((self.time_prediction_length, 20), dtype=np.float32)
        for t in range(self.time_prediction_length):
            target_states[t] = get_state(data, initial_x, initial_y, intial_time + t)

        return initial_state.unsqueeze(0), target_states


if __name__ == "__main__":
    path = r"C:\Users\tevch\Documents\Stage\Sentinel2TS\datasets\fontainebleau_interpolated"
    dataset = SentinelDataset(path)
    print('ok')
