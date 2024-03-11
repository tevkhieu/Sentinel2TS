import os
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.adam import Adam
from torch import nn
from sentinel2_ts.architectures.linear import Linear
import lightning as L


class LitLinear(L.LightningModule):
    """
    Lightning module for training a linear model representing the Koopman operator
    """

    def __init__(
        self,
        size: int,
        experiment_name: str,
        lr: int = 1e-3,
        time_span: int = 100,
        use_orthogonal_loss: bool = True,
        orthogonal_loss_weight: float = 10,
    ) -> None:
        super(LitLinear, self).__init__()
        self.k = Linear(size)
        self.time_span = time_span
        self.use_orthogonal_loss = use_orthogonal_loss
        self.experiment_name = experiment_name
        self.orthogonal_loss_weight = orthogonal_loss_weight
        self.val_loss = 1e5
        self.lr = lr
        self.save_dir = os.path.join("models", self.experiment_name)
        os.makedirs(self.save_dir, exist_ok=True)

    def forward(self, initial_state, time_span: int = 1) -> Tensor:
        return self.k(initial_state, time_span)

    def __compute_loss(
        self,
        phase: str,
        initial_state: Tensor,
        observed_states: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        predicted_time_series = self.forward(initial_state, self.time_span)[:, 1:, :]
        reconstruction_loss = self.__compute_reconstruction_loss(predicted_time_series, observed_states)
        loss_dict = {f"{phase} loss": reconstruction_loss}

        if self.use_orthogonal_loss:
            orthogonal_loss = self.orthogonal_loss_weight * self.__compute_orthogonal_loss()
            total_loss = reconstruction_loss + orthogonal_loss
            loss_dict[f"{phase} orthogonal loss"] = orthogonal_loss
            loss_dict[f"{phase} total loss"] = total_loss
            return total_loss, loss_dict

        return reconstruction_loss, loss_dict

    @staticmethod
    def __compute_reconstruction_loss(predicted_time_series, observed_states):
        criterion = nn.MSELoss()
        return criterion(predicted_time_series, observed_states)

    def __compute_orthogonal_loss(self):
        criterion = nn.MSELoss()
        k = self.k.k.weight
        return criterion(torch.matmul(k, k.T), torch.eye(k.size(0), device=self.device))

    def __save_k(self, loss):
        if loss < self.val_loss:
            self.val_loss = loss
            torch.save(self.k, os.path.join(self.save_dir, "best_k.pt"))

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
        return Adam(self.k.parameters(), lr=self.lr)
