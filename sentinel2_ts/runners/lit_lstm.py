import os
import torch
from torch import Tensor
from torch.optim import Optimizer, Adam
import torch.nn as nn
import lightning as L
from sentinel2_ts.architectures.lstm import LSTM


class LitLSTM(L.LightningModule):
    """Lightning module for training an LSTM"""

    def __init__(
        self,
        time_span: int,
        expermiment_name: str,
        lr: float = 2e-3,
    ) -> None:
        super().__init__()
        self.model = LSTM(20, 512, 20)
        self.criterion = nn.MSELoss()
        self.time_span = time_span
        self.lr = lr
        self.val_loss = 1e10
        self.experiment_name = expermiment_name
        self.save_dir = os.path.join("models", expermiment_name)

        os.makedirs(self.save_dir, exist_ok=True)

    def training_step(self, batch, batch_idx) -> Tensor:
        initial_state, observed_states = batch
        predicted_states = self.model(initial_state, self.time_span)
        loss = self.criterion(predicted_states, observed_states)
        self.log("train loss", loss)

        return loss

    def validation_step(self, batch, batch_idx) -> Tensor:
        initial_state, observed_states = batch
        predicted_states = self.model(initial_state, self.time_span)
        loss = self.criterion(predicted_states, observed_states)
        self.log("val loss", loss)
        if self.val_loss > loss:
            self.val_loss = loss
            torch.save(
                self.model.state_dict(),
                os.path.join(self.save_dir, f"best_{self.experiment_name}.pt"),
            )

        return loss

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.model.parameters(), lr=self.lr)

    def forward(self, x, time_span):
        return self.model(x, time_span)
