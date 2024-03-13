import torch
from torch import Tensor
import torch.nn as nn
from lightning import LightningModule
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader
from numpy.typing import NDArray
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class DataAssimilation:
    """Class for running data assimilation using a pre-trained model"""

    def __init__(
        self,
        log_dir: str,
        time_span: int = 241,
        lr: float = 1e-4,
        nb_epochs: int = 200,
        device: str = "cuda:0",
        nb_rows: int = 25,
        nb_columns: int = 100,
        initial_time: int = 1,
    ) -> None:
        self.time_span = time_span
        self.criterion = nn.MSELoss()
        self.lr = lr
        self.nb_epochs = nb_epochs
        self.device = device
        self.writer = SummaryWriter(log_dir=log_dir)
        self.nb_rows = nb_rows
        self.nb_columns = nb_columns
        self.initial_time = initial_time

    def data_assimilation(
        self,
        model: nn.Module,
        data: NDArray,
        range_x: tuple[int] = (0, 100),
        range_y: tuple[int] = (0, 100),
    ) -> Tensor:
        """
        Perform data assimilation on a grid of data

        Args:
            model (nn.Module): pretrained network
            data (NDArray): hyperspectral images time series
            range_x (tuple[int], optional): row indices. Defaults to (0, 100).
            range_y (tuple[int], optional): column indices. Defaults to (0, 100).
            nb_rows (int, optional): how many rows to use per batch. Defaults to 25.
            nb_columns (int, optional): how many columns to use per batch. Defaults to 100.

        Returns:
            assimilated_states (Tensor): initial states after data assimilation
        """
        x_min, x_max = range_x
        y_min, y_max = range_y
        len_x, len_y = x_max - x_min, y_max - y_min
        nb_rows = min(len_x, nb_rows)
        nb_columns = min(len_y, nb_columns)
        reflectance_grid = (data[..., x_min:x_max, y_min:y_max]).transpose(0, 2, 3, 1)
        assimilated_states = torch.zeros(len_x, len_y, 20, device="cpu")

        for rows in tqdm(range(len_x // nb_rows)):
            for columns in range(len_y // nb_columns):
                initial_state_grid = self.__extract_initial_state_grid(
                    reflectance_grid, rows, columns
                )

                observed_state_time_series = self.__extract_observied_time_series(
                    reflectance_grid, rows, columns
                )

                optimizer = Adam([initial_state_grid], lr=self.lr)
                for epoch in range(self.nb_epochs):
                    optimizer.zero_grad()
                    prediction = model(initial_state_grid, self.time_span)[:, 1:, :]
                    loss = self.criterion(prediction, observed_state_time_series)
                    loss.backward()
                    optimizer.step()
                    if epoch % 10 == 0:
                        self.writer.add_scalar(
                            f"loss batch row {rows} batch column {columns},",
                            loss,
                            epoch,
                        )

                assimilated_states[
                    rows * nb_rows : (rows + 1) * nb_rows,
                    columns * nb_columns : (columns + 1) * nb_columns,
                    :,
                ] = initial_state_grid.view(nb_rows, nb_columns, -1)

        return assimilated_states.cpu().detach().numpy()

    def __extract_initial_state_grid(
        self, reflectance_grid: NDArray, rows: int, columns: int
    ) -> Tensor:
        """
        Return the grid of data point of initial states

        Args:
            reflectance_grid (NDArray): full grid of reflectances
            rows (int): row index
            columns (int): column index

        Returns:
            Tensor: grid of initial states requiring grad
        """
        initial_reflectance = torch.clone(
            Tensor(
                reflectance_grid[
                    self.initial_time,
                    rows * self.nb_rows : (rows + 1) * self.nb_rows,
                    columns * self.nb_columns : (columns + 1) * self.nb_columns,
                    :,
                ]
            )
        )
        initial_reflectance_diff = torch.clone(
            Tensor(
                reflectance_grid[
                    self.initial_time,
                    rows * self.nb_rows : (rows + 1) * self.nb_rows,
                    columns * self.nb_columns : (columns + 1) * self.nb_columns,
                    :,
                ]
                - reflectance_grid[
                    self.initial_time - 1,
                    rows * self.nb_rows : (rows + 1) * self.nb_rows,
                    columns * self.nb_columns : (columns + 1) * self.nb_columns,
                    :,
                ]
            )
        )

        initial_state_grid = (
            torch.cat((initial_reflectance, initial_reflectance_diff), dim=2)
            .flatten(0, 1)
            .unsqueeze(1)
            .to(self.device)
        )
        initial_state_grid = initial_state_grid
        initial_state_grid = torch.clone(initial_state_grid).detach().requires_grad_()

        return initial_state_grid

    def __extract_observied_time_series(
        self, reflectance_grid: NDArray, rows: int, columns: int
    ) -> Tensor:
        """
        Generate time series of observed state

        Args:
            reflectance_grid (NDArray): full grid of reflectances
            rows (int): row index
            columns (int): column index

        Returns:
            Tensor: time series of observed states
        """
        observed_reflectance_time_series = Tensor(
            reflectance_grid[
                self.initial_time : self.time_span,
                rows * self.nb_rows : (rows + 1) * self.nb_rows,
                columns * self.nb_columns : (columns + 1) * self.nb_columns,
                :,
            ]
        )
        observed_reflectance_diff_time_series = Tensor(
            reflectance_grid[
                self.initial_time : self.time_span,
                rows * self.nb_rows : (rows + 1) * self.nb_rows,
                columns * self.nb_columns : (columns + 1) * self.nb_columns,
                :,
            ]
            - reflectance_grid[
                self.initial_time - 1 : self.time_span - 1,
                rows * self.nb_rows : (rows + 1) * self.nb_rows,
                columns * self.nb_columns : (columns + 1) * self.nb_columns,
                :,
            ]
        )
        observed_state_time_series = (
            torch.cat(
                (
                    observed_reflectance_time_series,
                    observed_reflectance_diff_time_series,
                ),
                dim=3,
            )
            .flatten(1, 2)
            .transpose(0, 1)
            .to(self.device)
        )
        return observed_state_time_series
