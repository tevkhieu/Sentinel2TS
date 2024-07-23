from torch.utils.data.dataloader import DataLoader
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from sentinel2_ts.dataset import (
    SentinelDataset,
    UnmixingDataset,
    TimeSeriesDataset,
    MultispectralImageDataset,
)


def get_dataloader(
    path: str,
    path_images: str = None,
    data_mode: str = None,
    time_span: int = 100,
    dataset_len: int = 512 * 512,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    minimal_x: int = 250,
    maximal_x: int = 400,
    minimal_y: int = 250,
    maximal_y: int = 400,
) -> DataLoader:
    """
    Create a Dataloader object from single point data

    Args:
        path (str): path to single point data folder
        time_span (int, optional): length of the target sequences. Defaults to 100.
        dataset_len (int, optional): number of data point used for training. Defaults to 512*512.
        batch_size (int, optional): batch size. Defaults to 64.
        shuffle (bool, optional): shuffle dataset for training. Defaults to True.
        num_workers (int, optional): number of workers. Defaults to 4.
        pin_memory (bool, optional): pin memory parameter. Defaults to True.

    Returns:
        DataLoader: Dataloader of single point data
    """
    match data_mode:
        case "spectral_disentangler":
            return DataLoader(
                UnmixingDataset(
                    path,
                    minimal_x=minimal_x,
                    maximal_x=maximal_x,
                    minimal_y=minimal_y,
                    maximal_y=maximal_y,
                ),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=True,
            )
        case "time_series":
            return DataLoader(
                TimeSeriesDataset(
                    path,
                    minimal_x=minimal_x,
                    maximal_x=maximal_x,
                    minimal_y=minimal_y,
                    maximal_y=maximal_y,
                ),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=True,
            )
        case "lucas_unmixer":
            return CombinedLoader(
                [
                    DataLoader(
                        MultispectralImageDataset(path_images),
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        persistent_workers=True,
                    ),
                    DataLoader(
                        TimeSeriesDataset(
                            path,
                            minimal_x=minimal_x,
                            maximal_x=maximal_x,
                            minimal_y=minimal_y,
                            maximal_y=maximal_y,
                        ),
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        persistent_workers=True,
                    ),
                ],
                mode="sequential",
            )

        case _:
            return DataLoader(
                SentinelDataset(
                    path,
                    time_prediction_length=time_span,
                    dataset_len=dataset_len,
                    minimal_x=minimal_x,
                    maximal_x=maximal_x,
                    minimal_y=minimal_y,
                    maximal_y=maximal_y,
                ),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=True,
            )
