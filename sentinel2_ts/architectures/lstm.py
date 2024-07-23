import torch
from torch import nn


class LSTM(nn.Module):
    """
    Classic LSTM architecture
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, future=0, n_steps=1):
        x = x.transpose(1, 2)
        batch_size = x.size(0)
        h_t = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        c_t = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        for i in range(x.size(1) - 1):
            out, (h_t, c_t) = self.lstm(x[:, i : i + 1, :], (h_t, c_t))
        out = self.fc(out)

        return out.squeeze(1)


if __name__ == "__main__":
    model = LSTM(20, 512, 4)
    x = torch.randn(1, 20, 342)
    print(model(x).shape)
