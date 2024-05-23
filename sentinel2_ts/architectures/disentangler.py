import torch
import torch.nn as nn
from sentinel2_ts.architectures.abundance_disentangler import AbundanceDisentangler
from sentinel2_ts.architectures.spectral_disentangler import SpectralDisentangler
from sentinel2_ts.architectures.lstm import LSTM


class Disentangler(nn.Module):
    """
    Time dependent endmember generator and time independant abundance generator

    """

    def __init__(self, size, latent_dim, num_classes, abundance_mode="conv") -> None:
        super().__init__()
        match abundance_mode:
            case "conv":
                self.abundance_disentangler = AbundanceDisentangler(size, num_classes)
            case "lstm":
                self.abundance_disentangler = LSTM(size, 512, num_classes)

        self.spectral_disentangler = SpectralDisentangler(size, latent_dim, num_classes)
        self.num_classes = num_classes
        self.size = size

    def forward(self, x):
        predicted_abundances = self.abundance_disentangler(x)
        predicted_specters = self.spectral_disentangler(x)
        predicted_specters = predicted_specters.view(
            predicted_specters.size(0), self.num_classes, self.size, -1
        )
        predicted_abundances = predicted_abundances.view(
            predicted_abundances.size() + (1, 1)
        )
        predicted_states = torch.sum(predicted_abundances * predicted_specters, dim=1)
        return predicted_states


if __name__ == "__main__":
    model = Disentangler(20, 64, 8, abundance_mode="lstm").to("cuda")
    x = torch.randn(256, 20, 342).to("cuda")
    predicted_specters = model.spectral_disentangler(x)
    predicted_specters = predicted_specters.view(predicted_specters.size(0), 8, 20, -1)
    print(predicted_specters.shape)

    predicted_abundances = model.abundance_disentangler(x)
    predicted_abundances = predicted_abundances.view(
        predicted_abundances.size() + (1, 1)
    )
    print(predicted_abundances.shape)
    predicted_states = torch.sum(predicted_abundances * predicted_specters, dim=1)
    print(predicted_states.shape)
