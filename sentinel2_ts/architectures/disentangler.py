import torch
import torch.nn as nn
from sentinel2_ts.architectures.abundance_disentangler import AbundanceDisentangler
from sentinel2_ts.architectures.spectral_disentangler import SpectralDisentangler
from sentinel2_ts.architectures.time_disentangler import TimeDisentangler
from sentinel2_ts.architectures.lstm import LSTM


class Disentangler(nn.Module):
    """
    Time dependent endmember generator and time independant abundance generator

    """

    def __init__(self, size, latent_dim, num_classes, abundance_mode="conv", disentangler_mode="specter") -> None:
        super().__init__()
        match abundance_mode:
            case "conv":
                self.abundance_disentangler = AbundanceDisentangler(size, num_classes)
            case "lstm":
                self.abundance_disentangler = LSTM(size, 512, num_classes)
        match disentangler_mode:
            case "specter":
                self.spectral_disentangler = SpectralDisentangler(size, latent_dim, num_classes)
            case "time":
                self.spectral_disentangler = TimeDisentangler(size, latent_dim, num_classes)
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

    def get_abundance(self, x):
        return self.abundance_disentangler(x) / (
            torch.sum(self.abundance_disentangler(x), dim=1, keepdim=True) + 1e-6
        )

    def forward_with_endmembers(self, x, endmembers, disentangler_mode="specter"):
        if disentangler_mode == "specter":
            predicted_abundances = self.abundance_disentangler(x)
            predicted_abundances = predicted_abundances.view(
                predicted_abundances.size() + (1, 1)
            )

            predicted_specters = self.spectral_disentangler(x)
            predicted_specters = predicted_specters.view(
                predicted_specters.size(0), self.num_classes, self.size, -1
            )
            modified_specters = predicted_specters.clone()
            modified_specters[:, :, :10, :] = modified_specters[
                :, :, :10, :
            ] * endmembers.view(1, endmembers.size(0), -1, 1)
            predicted_states = torch.sum(
                predicted_abundances * modified_specters, dim=1
            ).transpose(1, 2)
        else:
            predicted_abundances = self.abundance_disentangler(x)
            predicted_abundances = predicted_abundances.view(
                predicted_abundances.size() + (1, 1)
            )

            predicted_specters = self.spectral_disentangler(x)
            predicted_specters = predicted_specters.view(
                predicted_specters.size(0), self.num_classes, 1, -1
            ) * endmembers.view(1, endmembers.size(0), -1, 1)

            predicted_states = torch.sum(
                predicted_abundances * predicted_specters, dim=1
            )
        return predicted_states


if __name__ == "__main__":
    model = Disentangler(20, 64, 8, abundance_mode="lstm").to("cuda")
    x = torch.randn(256, 20, 342).to("cuda")
    endmembers = torch.randn(8, 10).to("cuda")

    print(model.forward_with_endmembers(x, endmembers).shape)

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
