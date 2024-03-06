import torch
from torch import Tensor
from torch.optim import Optimizer, Adam
import torch.nn as nn
import lightning as L
from sentinel2_ts.architectures.lstm import LSTM
from sentinel2_ts.architectures.koopman_ae import KoopmanAE

class LitLSTM(L.LightningModule):
    """Lightning module for training an LSTM"""

    def __init__(self, time_span: int) -> None:
        super().__init__()
        
        self.model = LSTM(20, 512, 20)
        self.criterion = nn.MSELoss()
        self.time_span = time_span

    def training_step(self, batch, batch_idx) -> Tensor:
        inputs, targets = batch
        outputs = self.model(inputs, self.time_span)
        loss = self.criterion(outputs, targets)
        self.log("train loss", loss)

        return loss

    def validation_step(self, batch, batch_idx) -> Tensor:
        inputs, targets = batch
        outputs = self.model(inputs, self.time_span)
        loss = self.criterion(outputs, targets)
        self.log("val loss", loss)

        return loss

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.model.parameters(), lr=2e-3)

    def forward(self, x):
        return self.model(x, self.time_span)
