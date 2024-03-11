import os
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.adam import Adam
from torch import nn
from sentinel2_ts.architectures.koopman_ae import KoopmanAE
import lightning as L


class LitKoopmanAE(L.LightningModule):
    """
    Lightning module for training a Koopman Auto-Encoder
    """

    def __init__(
        self,
        size: int,
        experiment_name: str,
        lr: int = 1e-3,
        time_span: int = 100,
        use_orthogonal_loss: bool = True,
        orthogonal_loss_weight: float = 10,
        device: str = "cuda:0"
    ) -> None:
        super(LitKoopmanAE, self).__init__()
        self.model = KoopmanAE(size, [512, 256, 32], device=device)
        self.time_span = time_span
        self.use_orthogonal_loss = use_orthogonal_loss
        self.experiment_name = experiment_name
        self.orthogonal_loss_weight = orthogonal_loss_weight
        self.val_loss = 1e5
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.save_dir = os.path.join("models", self.experiment_name)
        os.makedirs(self.save_dir, exist_ok=True)

    def __compute_loss(
        self,
        phase: str,
        initial_state: Tensor,
        observed_states: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        _, predicted_latent_time_series = self.model.forward_n_remember(initial_state, self.time_span)
        observed_latent_time_series = self.model.encode(observed_states)
        predicted_latent_time_series = predicted_latent_time_series[1:, :, 0, :].transpose(0, 1)
        reconstruction_loss = self.criterion(self.model.decode(predicted_latent_time_series), observed_states)

        decoding_loss = self.criterion(self.model.decode(observed_latent_time_series), observed_states)

        feature_loss = self.criterion(predicted_latent_time_series, observed_latent_time_series)


        orthogonality_loss = self.orthogonal_loss_weight * self.criterion(
            torch.matmul(self.model.K, self.model.K.T), torch.eye(self.model.K.size(0), device=self.device)
        )
        total_loss = reconstruction_loss + decoding_loss + feature_loss + orthogonality_loss

        loss_dict = {
            f"{phase} reconstruction loss": reconstruction_loss,
            f"{phase} decoding loss": decoding_loss,
            f"{phase} feature loss": feature_loss,
            f"{phase} orthogonality loss": orthogonality_loss,
            f"{phase} total loss": total_loss,
        }

        return total_loss, loss_dict

    def training_step(self, batch, batch_idx) -> Tensor:
        initial_state, observed_states = batch
        loss, loss_dict = self.__compute_loss("train", initial_state, observed_states)
        self.log_dict(loss_dict)
        return loss

    def validation_step(self, batch, batch_idx) -> Tensor:
        initial_state, observed_states = batch
        loss, loss_dict = self.__compute_loss("val", initial_state, observed_states)
        self.log_dict(loss_dict)
        self.__save_k(loss)
        return loss

    def configure_optimizers(self) -> Optimizer:
        """
        Configure the optimizer for training the model.

        Args:
            lr (float, optional): Learning rate for the optimizer (default: 1e-3).

        Returns:
            torch.optim.Optimizer: Optimizer instance.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        optimizer.add_param_group({"params": self.model.K})
        return optimizer

    def __save_k(self, loss):
        if loss < self.val_loss:
            self.val_loss = loss
            torch.save(self.model, os.path.join(self.save_dir, f"best_{self.experiment_name}.pt"))
