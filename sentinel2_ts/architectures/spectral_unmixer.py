import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralUnmixer(nn.Module):
    def __init__(self, x_range, y_range, size, latent_dim, num_classes) -> None:
        super().__init__()
        self.x_range = x_range
        self.y_range = y_range
        self.size = size
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(
            in_channels=size, out_channels=latent_dim[0], kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=latent_dim[0],
            out_channels=latent_dim[0],
            kernel_size=3,
            padding=1,
        )
        for i in range(1, len(latent_dim)):
            setattr(
                self,
                f"conv{i+1}",
                nn.Conv2d(
                    in_channels=latent_dim[i - 1],
                    out_channels=latent_dim[i],
                    kernel_size=3,
                    padding=1,
                ),
            )

        self.conv_out = nn.Conv2d(
            in_channels=latent_dim[-1],
            out_channels=size * num_classes,
            kernel_size=3,
            padding=1,
        )
        self.final_layer = nn.Linear(
            size
            * num_classes
            * (x_range // 2 ** (len(self.latent_dim)))
            * (y_range // 2 ** (len(self.latent_dim))),
            size * num_classes,
        )

    def forward(self, x):
        for i in range(len(self.latent_dim)):
            x = getattr(self, f"conv{i+1}")(x)
            x = torch.relu(x)
            x = F.max_pool2d(x, 2, 2)
        x = self.conv_out(x).flatten(1)
        x = self.final_layer(x)
        return x


if __name__ == "__main__":
    model = SpectralUnmixer(199, 199, 10, [16, 64], 4)
    x = torch.randn(1, 10, 199, 199)
    print(model(x).shape)
