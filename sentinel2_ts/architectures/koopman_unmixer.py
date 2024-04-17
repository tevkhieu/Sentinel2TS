import torch
import torch.nn as nn
import torch.nn.functional as F
from .sparsemax import Sparsemax


class KoopmanUnmixer(nn.Module):
    def __init__(self, input_dim: int, linear_dims: list, device="cpu"):
        """
        Koopman Unmixer class, comprising a non-linear encoder a Koopman matrix and a linear decoder.

        Args:
            input_dim (int): Dimension of the input data.
            linear_dims (list): List of linear layer dimensions.
            device (str, optional): Device to run the model on (default: 'cpu').
        """
        super(KoopmanUnmixer, self).__init__()

        self.latent_dim = linear_dims[-1]

        # Encoder layers
        self.encoder = nn.ModuleList()
        self.encoder.add_module("encoder_1", nn.Linear(input_dim, linear_dims[0]))
        for i in range(len(linear_dims) - 1):
            self.encoder.add_module(
                f"encoder_{i+2}", nn.Linear(linear_dims[i], linear_dims[i + 1])
            )
        # Koopman operator
        self.K = torch.eye(self.latent_dim, requires_grad=True, device=device)
        self.state_dict()["K"] = self.K

        self.decoder = nn.Linear(self.latent_dim, input_dim)
        self.final_activation = nn.ReLU()

    def encode(self, x):
        """Encode input data x using the encoder layers."""
        for layer_idx, layer in enumerate(self.encoder):
            x = layer(x)
            if layer_idx < len(self.encoder) - 1:
                x = F.relu(x)
        return x

    def one_step_ahead(self, x):
        """Predict one-step-ahead in the latent space using the Koopman operator."""
        return torch.matmul(x, self.K)

    def n_step_ahead(self, x, n):
        """Predict n-step-ahead in the latent space using the Koopman operator."""
        return torch.matmul(x, torch.matrix_power(self.K, n))

    def forward(self, x, time_span):
        """
        Perform forward pass through the model.

        Args:
            x (torch.Tensor): Input state.
            time_span (int): Number of steps to advance.

        Returns:
            predicted_states (torch.Tensor): Estimated states at each time step.
        """
        predicted_states = []
        phi = self.encode(x)
        for t in range(time_span - 1):
            phi_advanced = self.n_step_ahead(phi, t)
            predicted_states.append(self.decode(phi_advanced))
        return torch.stack(predicted_states, dim=1).squeeze(2)

    def decode(self, x):
        """Decode latent space representation x using the decoder layer."""
        return self.decoder(self.final_activation(x))

    def forward_n(self, x, n):
        """
        Perform forward pass for n steps.

        Args:
            x (torch.Tensor): Input state.
            n (int): Number of steps to advance.

        Returns:
            x_advanced (torch.Tensor): Estimated state after n time steps.
            phi (torch.Tensor): Encoded input state.
            phi_advanced (torch.Tensor): Encoded state advanced by n time steps.
        """
        phi = self.encode(x)
        phi_advanced = self.one_step_ahead(phi)
        for k in range(n - 1):
            phi_advanced = self.one_step_ahead(phi_advanced)
        x_advanced = self.decode(phi_advanced)
        return x_advanced, phi, phi_advanced

    def forward_n_remember(self, x, n, training=False):
        """
        Perform forward pass for n steps while remembering intermediate latent states.

        Args:
            x (torch.Tensor): Input state.
            n (int): Number of steps to advance.
            training (bool, optional): Flag to indicate training mode (default: False).

        Returns:
            x_advanced (torch.Tensor or None): Estimated state after n time steps if not training, otherwise None.
            phis (torch.Tensor): Encoded state at each step, concatenated along the 0th dimension.
        """
        phis = []
        phis.append(self.encode(x))
        for k in range(n):
            phis.append(self.one_step_ahead(phis[-1]))
        x_advanced = None if training else self.decode(phis[n])
        return x_advanced, torch.cat(tuple(phi.unsqueeze(0) for phi in phis), dim=0)

    def get_abundance_remember(self, x, n, eps: float = 1e-6):
        """
        Get abundance at each time step up to n.

        Args:
            x (torch.Tensor): Input state.
            n (int): Number of steps to advance.

        Returns:
            abundance_list (torch.Tensor): Abundances at each time step.
        """
        phi = self.encode(x)
        abundance_list = []
        for t in range(n - 1):
            phi = self.one_step_ahead(phi)
            abundance = self.final_activation(phi) / (
                torch.sum(self.final_activation(phi), dim=1, keepdim=True) + eps
            )
            abundance_list.append(abundance)
        return torch.stack(abundance_list, dim=1).squeeze(2)
