import os
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.adam import Adam
from torch import nn
from sentinel2_ts.architectures import SpectralDisentangler, AbundanceDisentangler
import lightning as L


class LitDisentangler(L.LightningModule):
    """
    Lightning module for training a spectral disentangler
    """

    def __init__(
        self,
        size: int,
        num_classes: int,
        experiment_name: str,
        lr: int = 1e-3,
    ) -> None:
        super(LitDisentangler, self).__init__()
        self.size = size
        self.num_classes = num_classes
        self.spectral_disentangler = SpectralDisentangler(size, num_classes=num_classes)
        self.abundance_disentangler = AbundanceDisentangler(
            size, num_classes=num_classes
        )
        self.experiment_name = experiment_name
        self.val_loss = 1e5
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.kl_div_loss = nn.KLDivLoss()
        self.save_dir = os.path.join("models", self.experiment_name)
        os.makedirs(self.save_dir, exist_ok=True)

    def __compute_loss(
        self,
        phase: str,
        observed_states: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        z, mu, sigma = self.spectral_disentangler.encode(observed_states)
        predicted_specters = self.spectral_disentangler.decode(z.view(z.size(0), 64, -1))
        predicted_specters = predicted_specters.view(
            predicted_specters.size(0), self.num_classes, self.size, -1
        )
        predicted_abundances = self.abundance_disentangler(observed_states)
        predicted_abundances = predicted_abundances.view(
            predicted_abundances.size() + (1, 1)
        )

        predicted_states = torch.sum(predicted_abundances * predicted_specters, dim=1)

        recon_loss = self.criterion(predicted_states, observed_states)
        kld_loss = 150 * torch.mean(
            -0.5 * torch.sum(1 + sigma - mu**2 - sigma.exp(), dim=1), dim=0
        )

        total_loss = recon_loss + kld_loss
        loss_dict = {
            f"{phase} recon_loss": recon_loss,
            f"{phase} kld_loss": kld_loss,
            f"{phase} total_loss": total_loss,
        }

        return total_loss, loss_dict

    def training_step(self, batch, batch_idx) -> Tensor:
        observed_states = batch
        loss, loss_dict = self.__compute_loss("train", observed_states)
        self.log_dict(loss_dict)
        return loss

    def validation_step(self, batch, batch_idx) -> Tensor:
        observed_states = batch
        loss, loss_dict = self.__compute_loss("val", observed_states)
        self.log_dict(loss_dict)
        self.__save(loss)
        return loss

    def configure_optimizers(self) -> Optimizer:
        """
        Configure the optimizer for training the model.

        Args:
            lr (float, optional): Learning rate for the optimizer (default: 1e-3).

        Returns:
            torch.optim.Optimizer: Optimizer instance.
        """
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer

    def __save(self, loss):
        if loss < self.val_loss:
            self.val_loss = loss
            torch.save(
                self.spectral_disentangler.state_dict(),
                os.path.join(self.save_dir, f"best_spectral_disentangler.pt"),
            )
            torch.save(
                self.abundance_disentangler.state_dict(),
                os.path.join(self.save_dir, f"best_abundance_disentangler.pt"),
            )
