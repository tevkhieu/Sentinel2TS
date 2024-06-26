import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class ResidualSpecter:
    """
    Learn the initial abundances and the time variations of the endmembers via automatic differentiation
    """

    def __init__(
        self,
        data: np.ndarray,
        endmembers: np.ndarray,
        device: str = "cuda",
        lr: float = 1e-3,
    ) -> None:
        """

        Args:
            data (np.ndarray): map of reflectance time series
            endmembers (np.ndarray): endmembers of the spectral unmixing
            device (str, optional): cuda | cpu. Defaults to "cuda".
            lr (float, optional): learning rate. Defaults to 1e-3.
        """
        self.device = device
        self.criterion = torch.nn.MSELoss()
        self.lr = lr
        self.writer = SummaryWriter(log_dir="logs")
        self.data = torch.tensor(data.transpose(0, 2, 3, 1)).float().to(self.device)
        self.initial_abundance_map = torch.nn.Parameter(
            torch.randn((1, 500, 500, 5, 1), requires_grad=True, device=self.device)
        )
        self.endmembers = (
            torch.tensor(endmembers).float().view(1, 1, 1, 5, 10).to(self.device)
        )
        self.initial_variation_map = torch.nn.Parameter(
            torch.zeros((343, 500, 500, 5, 1), requires_grad=True, device=self.device)
        )

    def __train_step_one(self) -> None:
        """
        Learn the initial abundance map
        """
        optimizer = torch.optim.Adam([self.initial_abundance_map], lr=self.lr)

        for i in tqdm(range(40000)):
            optimizer.zero_grad()
            loss = self.criterion(
                self.data[0],
                torch.sum(
                    torch.exp(self.initial_abundance_map[0]) * self.endmembers[0], dim=2
                ),
            )
            if i % 1000 == 0:
                self.writer.add_scalar("Abundance Loss", loss.item(), i)
            loss.backward()
            optimizer.step()

    def __train_step_two(self) -> None:
        """
        Learn the time variations of the endmembers
        """
        optimizer = torch.optim.Adam([self.initial_variation_map], lr=self.lr)

        for i in tqdm(range(20)):
            for x in range(500):
                optimizer.zero_grad()
                variations = torch.sum(
                    torch.exp(self.initial_abundance_map)[:, x]
                    * self.initial_variation_map[:, x]
                    * self.endmembers[:, 0],
                    dim=2,
                )
                loss = self.criterion(self.data[:, x, :], variations)
                if x == 0:
                    self.writer.add_scalar("Variation Loss", loss, i)
                loss.backward()
                optimizer.step()

    def train(self):
        """
        Learn the weights of the model
        """
        self.__train_step_one()
        self.__train_step_two()
        np.save(
            "initial_abundance_map.npy",
            self.initial_abundance_map[0, :, :, :, 0].detach().cpu().numpy(),
        )
        np.save(
            "initial_variation_map.npy",
            self.initial_variation_map[:, :, :, :, 0].detach().cpu().numpy(),
        )
        self.writer.close()
