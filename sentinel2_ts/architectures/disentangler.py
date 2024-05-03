import torch.nn as nn
from sentinel2_ts.architectures.abundance_disentangler import AbundanceDisentangler
from sentinel2_ts.architectures.spectral_disentangler import SpectralDisentangler


class Disentangler(nn.Module):
    def __init__(self, size, latent_dim, num_classes) -> None:
        super().__init__()
        self.spectral_disentangler = SpectralDisentangler(size, latent_dim, num_classes)
        self.abundance_disentangler = AbundanceDisentangler(size, num_classes)

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
