import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralDisentangler(nn.Module):
    def __init__(self, size: int = 20, latent_dim: int = 64, num_classes: int = 5):
        super(SpectralDisentangler, self).__init__()
        self.conv1 = nn.Conv1d(size, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)

        self.fc_mu = nn.Linear(64 * (342 // 4), latent_dim * (342 // 4))
        self.fc_sigma = nn.Linear(64 * (342 // 4), latent_dim * (342 // 4))

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(
            128, size * num_classes, kernel_size=2, stride=1, padding=1
        )
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(size * num_classes)

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

        # z = z.view(z.size(0), -1)

        # mu, sigma = self.fc_mu(z), self.fc_sigma(z)
        # z = self.reparametrize(mu, sigma)

        return z

    def reparametrize(self, mu, sigma):
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        return mu + eps * std


if __name__ == "__main__":
    import torch

    model = SpectralDisentangler()
    model_input = torch.randn(512, 20, 342)
    z = model.encode(model_input)
    print(z.shape)

    decoded = model.decode(z.view(512, 64, -1))
    print(decoded.shape)
