import os
import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.adam import Adam
from torch import nn
from sentinel2_ts.architectures import Disentangler
import lightning as L


class LitDisentangler(L.LightningModule):
    """
    Lightning module for training a disentangler
    """

    def __init__(
        self,
        size: int,
        latent_dim: int,
        num_classes: int,
        experiment_name: str,
        endmembers: np.ndarray = None,
        abundance_mode: str = "conv",
        lr: int = 1e-5,
        beta: float = 150.0,
        disentangler_mode: str = "specter",

    ) -> None:
        """
        Initialize the Lightning module

        Args:
            size (int): Size of the input data
            latent_dim (int): Dimension of the latent space
            num_classes (int): Number of classes
            experiment_name (str): name of the folder to save the model
            endmembers (np.ndarray, optional): array of endmember if the specters are not learnt. Defaults to None.
            abundance_mode (str, optional): conv | lstm. Defaults to "conv".
            lr (int, optional): learning rate. Defaults to 1e-5.
            beta (float, optional): weight for beta vae. Defaults to 150.0.
        """
        super(LitDisentangler, self).__init__()
        self.size = size
        self.latent_dim = latent_dim
        if endmembers is not None:
            self.endmembers = (
                torch.tensor(endmembers)
                .float()
                .view(1, endmembers.shape[0], -1, 1)
                .to("cuda")
            )
        else:
            self.endmembers = None
        self.num_classes = (
            num_classes if endmembers is None else self.endmembers.shape[1]
        )
        self.beta = beta
        self.disentangler_mode = disentangler_mode
        self.model = Disentangler(size, latent_dim, self.num_classes, abundance_mode, disentangler_mode)
        self.experiment_name = experiment_name
        self.val_loss = 1e5
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.save_dir = os.path.join("models", self.experiment_name)
        os.makedirs(self.save_dir, exist_ok=True)

    def __compute_loss(
        self,
        phase: str,
        observed_states: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Compute the loss of the model

        Args:
            phase (str): train | val
            observed_states (Tensor): observed states

        Returns:
            tuple[Tensor, dict[str, Tensor]]: total loss and dictionary of losses
        """

        predicted_specters = self.model.spectral_disentangler(observed_states)
        predicted_abundances = self.model.abundance_disentangler(observed_states)
        predicted_abundances = predicted_abundances.view(
            predicted_abundances.size() + (1, 1)
        )

        if self.disentangler_mode == "specter":
            predicted_specters = predicted_specters.view(
                predicted_specters.size(0), self.num_classes, self.size, -1
            )
        else:
            predicted_specters = predicted_specters.view(
                predicted_specters.size(0), self.num_classes, 1, -1
            ) * self.endmembers
        loss_dict = {}
        
        if self.endmembers is not None:
            if self.disentangler_mode == "specter":
                modified_specters = predicted_specters.clone()
                modified_specters[:, :, : self.size // 2, :] = (
                    modified_specters[:, :, : self.size // 2, :] * self.endmembers
                )
                predicted_states = torch.sum(
                    predicted_abundances * modified_specters, dim=1
                ).transpose(1, 2)
                # predicted_states = torch.sum(
                #     predicted_abundances * predicted_specters, dim=1
                # ).transpose(1, 2)

                # loss_dict = self.__compute_cosine_loss(phase, loss_dict, predicted_specters)
            else:
                predicted_states = torch.sum(
                    predicted_abundances * predicted_specters, dim=1
                ).transpose(1, 2)

        else:
            predicted_states = torch.sum(
                predicted_abundances * predicted_specters, dim=1
            )

        loss_dict = self.__compute_sparsity_loss(phase, predicted_abundances, loss_dict)
        loss_dict = self.__compute_recon_loss(
            phase, predicted_states, observed_states, loss_dict
        )

        total_loss, loss_dict = self.__compute_total_loss(phase, loss_dict)

        return total_loss, loss_dict

    def training_step(self, batch, batch_idx) -> Tensor:
        if self.disentangler_mode == "time":
            observed_states, _, _ = batch
        else:
            observed_states = batch
        loss, loss_dict = self.__compute_loss("train", observed_states)
        self.log_dict(loss_dict)
        return loss

    def validation_step(self, batch, batch_idx) -> Tensor:
        if self.disentangler_mode == "time":
            observed_states, _, _ = batch
        else:
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
                self.model.state_dict(),
                os.path.join(self.save_dir, f"best_{self.experiment_name}.pt"),
            )

    def __compute_sparsity_loss(
        self, phase, predicted_abundances, loss_dict: dict[str, Tensor] = None
    ):
        loss_dict[f"{phase}_sparsity_loss"] = 1e-2 * torch.mean(
            torch.abs(predicted_abundances)
        )
        return loss_dict

    def __compute_recon_loss(
        self,
        phase,
        predicted_states,
        observed_states,
        loss_dict: dict[str, Tensor] = None,
    ):
        loss_dict[f"{phase}_recon_loss"] = 10 * self.criterion(
            predicted_states, observed_states.transpose(1, 2)
        )
        return loss_dict

    def __compute_total_loss(self, phase, loss_dict: dict[str, Tensor]) -> Tensor:
        loss_dict[f"{phase}_total_loss"] = 0
        for key in loss_dict:
            loss_dict[f"{phase}_total_loss"] += loss_dict[key]

        return loss_dict[f"{phase}_total_loss"], loss_dict

    def __compute_cosine_loss(self, phase, loss_dict, predicted_specters):
        endmembers = self.endmembers.expand(
            (predicted_specters.size(0), -1, -1, predicted_specters.size(3))
        ).to(self.device)
        cosine_loss = 1 - 1e-3 * torch.nn.functional.cosine_similarity(
            endmembers, predicted_specters[:, :, : self.size // 2, :], dim=2
        )

        mse_loss = torch.nn.functional.mse_loss(
            predicted_specters[:, :, : self.size // 2, :], endmembers
        )

        loss_dict[f"{phase}_cosine_loss"] = torch.mean(cosine_loss) + 1e-3 * mse_loss

        return loss_dict
