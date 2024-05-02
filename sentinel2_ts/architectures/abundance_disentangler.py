import torch.nn as nn
import torch.nn.functional as F


class AbundanceDisentangler(nn.Module):
    def __init__(self, input_size=20, num_classes=5):
        super(AbundanceDisentangler, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(64 * (342 // 4), num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = F.leaky_relu(out)
        out = F.max_pool1d(out, kernel_size=2, stride=2)

        out = self.conv2(out)
        out = F.leaky_relu(out)
        out = F.max_pool1d(out, kernel_size=2, stride=2)

        out = out.view(out.size(0), -1)
        out = F.relu(self.fc(out))

        return out

if __name__ == "__main__":
    import torch
    model = AbundanceDisentangler()
    x = torch.randn(512, 20, 342)
    print(model(x).shape)