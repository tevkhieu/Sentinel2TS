import os
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.adam import Adam
from torch import nn
import torch.nn.functional as F
from sentinel2_ts.architectures import KoopmanUnmixer
import lightning as L


class LitKoopmanUnmixer(L.LightningModule):
    """
    Lightning module for training a Koopman Auto-Encoder
    """

    def __init__(
        self,
        size: int,
        experiment_name: str,
        latent_dim: list[int] = [512, 256, 32],
        lr: int = 1e-5,
        time_span: int = 100,
        use_orthogonal_loss: bool = True,
        orthogonal_loss_weight: float = 10,
        device: str = "cuda:0",
    ) -> None:
        super(LitKoopmanUnmixer, self).__init__()
        self.model = KoopmanUnmixer(size, latent_dim, device=device)
        self.time_span = time_span
        self.use_orthogonal_loss = use_orthogonal_loss
        self.experiment_name = experiment_name
        self.orthogonal_loss_weight = orthogonal_loss_weight
        self.val_loss = 1e5
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.save_dir = os.path.join("models", self.experiment_name)
        os.makedirs(self.save_dir, exist_ok=True)

        self.automatic_optimization = False

    def __compute_loss(
        self,
        phase: str,
        initial_state: Tensor,
        observed_states: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        _, predicted_latent_time_series = self.model.forward_n_remember(
            initial_state, self.time_span
        )
        observed_latent_time_series = self.model.encode(observed_states)
        predicted_latent_time_series = predicted_latent_time_series[
            1:, :, 0, :
        ].transpose(0, 1)
        abundance_time_series = self.model.decode_abundance(
            predicted_latent_time_series
        )
        reconstruction_time_series = self.model.final_layer(abundance_time_series)

        loss_dict = {}

        _, loss_dict = self.__compute_reconstruction_loss(
            phase, loss_dict, reconstruction_time_series, observed_states
        )

        _, loss_dict = self.__compute_decoding_loss(
            phase, loss_dict, observed_latent_time_series, observed_states
        )

        _, loss_dict = self.__compute_feature_loss(
            phase, loss_dict, predicted_latent_time_series, observed_latent_time_series
        )

        _, loss_dict = self.__compute_orthogonality_loss(phase, loss_dict)

        _, loss_dict = self.__compute_sparsity_loss(
            phase, loss_dict, abundance_time_series
        )

        total_loss, loss_dict = self.__compute_total_loss(phase, loss_dict)

        return total_loss, loss_dict

    def training_step(self, batch, batch_idx) -> Tensor:
        initial_state, observed_states = batch
        opt = self.optimizers()
        opt.zero_grad()
        loss, loss_dict = self.__compute_loss("train", initial_state, observed_states)
        self.log_dict(loss_dict)
        self.manual_backward(loss)
        opt.step()
        with torch.no_grad():
            self.model.final_layer.weight[:10, :].copy_(
                self.model.final_layer.weight[:10, :].clamp(min=0)
            )

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
        optimizer = Adam(self.parameters(), lr=self.lr)
        optimizer.add_param_group({"params": self.model.K})
        return optimizer

    def __save_k(self, loss):
        if loss < self.val_loss:
            self.val_loss = loss
            torch.save(
                self.model.state_dict(),
                os.path.join(self.save_dir, f"best_{self.experiment_name}.pt"),
            )
            torch.save(
                self.model.K,
                os.path.join(self.save_dir, f"best_k.pt"),
            )

    def __compute_sparsity_loss(self, phase, loss_dict, abundance_time_series):
        sparsity_loss = 1e-1 * F.l1_loss(
            abundance_time_series, torch.zeros_like(abundance_time_series)
        )
        loss_dict[f"{phase} sparsity loss"] = sparsity_loss
        return sparsity_loss, loss_dict

    def __compute_reconstruction_loss(
        self, phase, loss_dict, reconstruction_time_series, observed_states
    ):
        reconstruction_loss = self.criterion(
            reconstruction_time_series, observed_states
        )
        loss_dict[f"{phase} reconstruction loss"] = reconstruction_loss
        return reconstruction_loss, loss_dict

    def __compute_decoding_loss(
        self, phase, loss_dict, observed_latent_time_series, observed_states
    ):
        decoding_loss = self.criterion(
            self.model.decode(observed_latent_time_series), observed_states
        )
        loss_dict[f"{phase} decoding loss"] = decoding_loss
        return decoding_loss, loss_dict

    def __compute_feature_loss(
        self,
        phase,
        loss_dict,
        predicted_latent_time_series,
        observed_latent_time_series,
    ):
        feature_loss = self.criterion(
            predicted_latent_time_series, observed_latent_time_series
        )
        loss_dict[f"{phase} feature loss"] = feature_loss
        return feature_loss, loss_dict

    def __compute_orthogonality_loss(self, phase, loss_dict):
        orthogonality_loss = self.orthogonal_loss_weight * self.criterion(
            torch.matmul(self.model.K, self.model.K.T),
            torch.eye(self.model.K.size(0), device=self.device),
        )
        loss_dict[f"{phase} orthogonality loss"] = orthogonality_loss
        return orthogonality_loss, loss_dict

    def __compute_total_loss(self, phase, loss_dict):
        total_loss = sum(loss_dict.values())
        loss_dict[f"{phase} total loss"] = total_loss
        return total_loss, loss_dict
