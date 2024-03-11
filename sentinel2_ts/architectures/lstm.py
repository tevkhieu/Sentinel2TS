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
        outputs = []
        batch_size = x.size(0)
        h_t = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        c_t = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        # Iterate through time steps
        for i in range(future):
            out, (h_t, c_t) = self.lstm(x[:, -1:, :], (h_t, c_t))
            out = self.fc(out)
            outputs.append(out)
            x = torch.cat((x, out), dim=1)

        outputs = torch.cat(outputs, dim=1)
        return outputs
