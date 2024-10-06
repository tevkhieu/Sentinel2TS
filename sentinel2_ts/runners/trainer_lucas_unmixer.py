import torch
import os
import numpy as np
from torch.optim.adam import Adam
from torch import Tensor
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data.dataloader import DataLoader
from lightning.pytorch.utilities.combined_loader import CombinedLoader

from sentinel2_ts.architectures.lucas_unmixer import LucasUnmixer
from sentinel2_ts.dataset import MultispectralImageDataset, TimeSeriesDataset


class TrainerLucasUnmixer:
    """
    Module to train the Lucas Unmixer model.
    """

    def __init__(
        self,
        num_classes,
        experiment_name,
        size,
        latent_dim,
        image_dataset_path: str,
        time_series_dataset_path: str,
        time_range: int = 342,
        lr=1e-3,
        device="cuda:0",
        batch_size=1,
        minimal_x=0,
        maximal_x=199,
        minimal_y=0,
        maximal_y=199,
        endmembers_path=None,
    ) -> None:
        self.size = size
        self.num_classes = num_classes
        self.time_range = time_range
        self.x_range = maximal_x - minimal_x
        self.y_range = maximal_y - minimal_y
        self.automatic_optimization = False
        self.lr = lr
        self.save_dir = os.path.join("models", experiment_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.experiment_name = experiment_name
        self.model = LucasUnmixer(
            size=size,
            latent_dim=latent_dim,
            num_classes=num_classes,
            x_range=self.x_range,
            y_range=self.y_range,
        ).to(device)
        lightning_ckpt_path = os.path.join("models_lightning_ckpt", experiment_name)
        if os.path.exists(lightning_ckpt_path):
            version_path = os.path.join(
                lightning_ckpt_path, f"version_{len(os.listdir(lightning_ckpt_path))}"
            )
            os.makedirs(version_path, exist_ok=True)
        else:
            version_path = os.path.join(lightning_ckpt_path, "version_0")
            os.makedirs(version_path, exist_ok=True)
        self.writer = SummaryWriter(log_dir=version_path)

        self.predicted_specters = torch.ones(
            1, num_classes, time_range, size, requires_grad=True
        ).to(
            device
        )  # (batch_size, num_classes, time_range, size)
        self.predicted_abundances = (
            torch.ones(
                1, num_classes, 1, self.x_range, self.y_range, requires_grad=True
            )
            / num_classes
        ).to(
            device
        )  # (batch_size, num_classes, size, x_range, y_range)
        self.image_dataset_path = image_dataset_path
        self.time_series_dataset_path = time_series_dataset_path
        self.optimizers = self.__configure_optimizers()
        self.batch_size = batch_size
        self.minimal_x = minimal_x
        self.maximal_x = maximal_x
        self.minimal_y = minimal_y
        self.maximal_y = maximal_y
        self.device = device
        self.data = np.load("data/fontainebleau_interpolated.npy")[:, :, minimal_x:maximal_x, minimal_y:maximal_y]
        self.endmembers = torch.Tensor(np.load(endmembers_path))

    @torch.no_grad()
    def __update_predicted_specters(self):
        for t in range(self.time_range):
            image = self.data[t, :, :, :]
            image = torch.Tensor(image).to(self.device).unsqueeze(0)
            self.predicted_specters[:, :, t, :] = self.model.get_specter(image)

    @torch.no_grad()
    def __update_predicted_abundances(self):
        for x in (range(self.x_range)):
            time_series = self.data[:, :, x, :]
            time_series = torch.Tensor(time_series).to(self.device).transpose(0, 2)
            self.predicted_abundances[0, :, 0, x, :] = self.model.get_abundance(time_series).transpose(0, 1)

    def __configure_optimizers(self):
        """
        Configure the optimizer for training the model.

        Args:
            lr (float, optional): Learning rate for the optimizer (default: 1e-3).

        Returns:
            torch.optim.Optimizer: Optimizer instance.
        """
        optimizer_abundances = Adam(
            self.model.abundance_unmixer.parameters(), lr=self.lr
        )
        optimizer_specters = Adam(self.model.spectral_unmixer.parameters(), lr=self.lr)
        return optimizer_abundances, optimizer_specters

    def __training_step(self, batch, dataloader_idx) -> Tensor:
        loss = self.__compute_loss("train", batch, dataloader_idx)

        return loss

    def __update(self):
            self.__update_predicted_specters()
            self.__update_predicted_abundances()

    def __save(self):
        torch.save(
            self.model.state_dict(),
            os.path.join(self.save_dir, f"best_{self.experiment_name}.pt"),
        )
        np.save(
            os.path.join(self.save_dir, f"best_specters_{self.experiment_name}.pt"),
            self.predicted_specters[0].cpu().detach().numpy(),
        )
        np.save(
            os.path.join(self.save_dir, f"best_abundances_{self.experiment_name}.npy"),
            self.predicted_abundances[0, :, 0].cpu().detach().numpy().transpose(1, 2, 0),
        )

    def __compute_loss(self, phase, batch, dataloader_idx):
        loss_dict = {}

        if dataloader_idx == 0:
            batch, _ = batch
            loss, loss_dict = self.__compute_specter_extractor_loss(
                phase, batch, loss_dict
            )

        else:
            batch, _, _ = batch
            loss, loss_dict = self.__compute_abundance_extractor_loss(
                phase, batch, loss_dict
            )
        return loss

    def __compute_specter_extractor_loss(self, phase, batch, loss_dict):
        multispectral_image = batch
        multispectral_image = multispectral_image.to(self.device)
        predicted_specters = self.model.get_specter(multispectral_image)
        predicted_specters = predicted_specters.view(
            predicted_specters.size(0), self.num_classes, self.size, 1, 1
        )
        predicted_image = torch.sum(
            self.predicted_abundances * predicted_specters, dim=1
        )
        loss_specter = self.__compute_cosine_loss(predicted_specters)
        loss_image = torch.mean((multispectral_image - predicted_image) ** 2)
        loss = loss_specter + loss_image

        return loss, loss_dict

    def __compute_abundance_extractor_loss(self, phase, batch, loss_dict):
        abundance_time_series = batch
        abundance_time_series = abundance_time_series.to(self.device)
        predicted_abundances = self.model.get_abundance(abundance_time_series)
        predicted_abundances = predicted_abundances.view(
            predicted_abundances.size() + (1, 1)
        )
        predicted_image = torch.sum(
            predicted_abundances * self.predicted_specters, dim=1
        )

        loss = torch.mean(
            (abundance_time_series - predicted_image.transpose(1, 2)) ** 2
        )

        return loss, loss_dict

    def __configure_dataloader(self):
        return CombinedLoader(
            [
                DataLoader(
                    MultispectralImageDataset(self.image_dataset_path),
                    batch_size=32,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True,
                    persistent_workers=True,
                ),
                DataLoader(
                    TimeSeriesDataset(
                        self.time_series_dataset_path,
                        minimal_x=self.minimal_x,
                        maximal_x=self.maximal_x,
                        minimal_y=self.minimal_y,
                        maximal_y=self.maximal_y,
                    ),
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True,
                    persistent_workers=True,
                ),
            ],
            mode="sequential",
        )

    def train(self, max_epochs):
        train_dataloader = self.__configure_dataloader()

        for epoch in tqdm(range(max_epochs)):
            for batch, batch_idx, dataloader_idx in train_dataloader:
                loss = self.__training_step(batch, dataloader_idx)
                if not batch_idx % 10:
                    if dataloader_idx == 0:
                        self.writer.add_scalar(
                            "spectral loss",
                            loss,
                            epoch * len(train_dataloader) + batch_idx,
                        )
                    else:
                        self.writer.add_scalar(
                            "abundance loss",
                            loss,
                            epoch * len(train_dataloader) + batch_idx,
                        )
                if dataloader_idx == 0:
                    _, optimizer_spectral = self.optimizers
                    optimizer_spectral.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer_spectral.step()
                else:
                    optimizer_abundances, _ = self.optimizers
                    optimizer_abundances.zero_grad()
                    loss.backward()
                    optimizer_abundances.step()
            self.__update()
            self.__save()

        self.writer.close()

    def __compute_cosine_loss(self, predicted_specters):
        endmembers = self.endmembers.expand(predicted_specters.size(0), self.num_classes, self.size)
        endmembers = endmembers.view(predicted_specters.size(0), self.num_classes, self.size, 1, 1).to(self.device)

        mse_loss = torch.nn.functional.mse_loss(
            predicted_specters, endmembers
        )

        cosine_loss = torch.nn.functional.cosine_similarity(
            predicted_specters, endmembers, dim=2
        )        

        return 1e-2 * mse_loss + 1 - torch.mean(cosine_loss)
