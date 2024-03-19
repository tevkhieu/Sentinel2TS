import torch
import torch.nn as nn
from torch import Tensor


class Linear(nn.Module):
    """Linear network learning Koopman operator"""

    def __init__(self, size: int = 20) -> None:
        super().__init__()

        self.k = nn.Linear(size, size, bias=False)
        self.k.weight.data.copy_(torch.eye(size))

    def forward(self, initial_state: Tensor, time_span: int):
        predicted_states = [initial_state.squeeze()]
        for _ in range(time_span):
            predicted_states.append(self.k(predicted_states[-1]))

        return torch.stack(predicted_states[1:], dim=0)


if __name__ == "__main__":
    model = Linear(20)
    x = torch.rand(1, 1, 20)
    y = model(x, 100)
    print(y.shape)
