import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeDisentangler(nn.Module):
    def __init__(self, size: int = 20, latent_dim: int = 64, num_classes: int = 5):
        super(TimeDisentangler, self).__init__()
        self.conv1 = nn.Conv1d(size, 32, kernel_size=3, stride=1, padding=1, padding_mode="circular")
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, padding_mode="circular")
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, padding_mode="circular")
        self.conv4 = nn.Conv1d(
            128, num_classes, kernel_size=2, stride=1, padding=1, padding_mode="circular"
        )
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(num_classes)

    def forward(self, x):
        z = self.encode(x)

        out = self.decode(z.reshape(z.size(0), 64, -1))

        return out

    def decode(self, out):
        out = self.conv3(out)
        out = self.bn3(out)
        out = F.interpolate(out, scale_factor=2)
        out = F.leaky_relu(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = F.interpolate(out, scale_factor=2)
        out = F.softplus(out)

        return out

    def encode(self, x):
        z = self.conv1(x)
        z = self.bn1(z)
        z = F.leaky_relu(z)
        z = F.max_pool1d(z, kernel_size=2, stride=2)

        z = self.conv2(z)
        z = self.bn2(z)
        z = F.leaky_relu(z)
        z = F.max_pool1d(z, kernel_size=2, stride=2)

        return z


if __name__ == "__main__":
    import torch

    model = SpectralDisentangler()
    model_input = torch.randn(512, 20, 342)
    z = model.encode(model_input)
    print(z.shape)

    decoded = model.decode(z.view(512, 64, -1))
    print(decoded.shape)
