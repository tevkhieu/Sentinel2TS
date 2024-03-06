from torch.utils.data.dataloader import DataLoader

from sentinel2_ts.utils.sentinel_dataset import SentinelDataset

def get_dataloader(path: str, time_span: int = 100, dataset_len: int = 512 * 512, batch_size: int = 64, shuffle: bool = True, num_workers: int = 4, pin_memory: bool = True):
    return DataLoader(
        SentinelDataset(path, time_prediction_length=time_span, dataset_len=dataset_len),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )