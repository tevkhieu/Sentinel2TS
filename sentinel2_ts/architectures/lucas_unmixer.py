import torch
import torch.nn as nn
from sentinel2_ts.architectures.spectral_unmixer import SpectralUnmixer
from sentinel2_ts.architectures.lstm import LSTM


class LucasUnmixer(nn.Module):
    """
    Lucas Unmixer model
    """

    def __init__(self, size, num_classes, latent_dim, x_range, y_range) -> None:
        super().__init__()
        self.size = size
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.x_range = x_range
        self.y_range = y_range
        self.abundance_unmixer = LSTM(size, 512, self.num_classes)
        self.spectral_unmixer = SpectralUnmixer(
            x_range, y_range, size, latent_dim, num_classes
        )

    def get_specter(self, multispectral_image):
        return self.spectral_unmixer(multispectral_image)

    def get_abundance(self, time_series):
        return self.abundance_unmixer(time_series)

    def get_normalized_abundance(self, time_series):
        return self.abundance_unmixer(time_series) / (
            torch.sum(self.abundance_unmixer(time_series), dim=1, keepdim=True) + 1e-6
        )

    def forward(self, time_series, multispectral_image):
        predicted_abundance = self.abundance_unmixer(time_series).view(
            predicted_abundance.size() + (1,)
        )  # (batch_size, num_classes, 1)
        predicted_specter = self.spectral_unmixer(multispectral_image).view(
            predicted_specter.size(0), self.num_classes, self.size
        )  # (batch_size, num_classes, size)
        predicted_state = torch.sum(predicted_abundance * predicted_specter, dim=1)

        return predicted_state


if __name__ == "__main__":
    model = LucasUnmixer(
        size=199, latent_dim=[20, 64, 128], num_classes=4, x_range=199, y_range=199
    )
