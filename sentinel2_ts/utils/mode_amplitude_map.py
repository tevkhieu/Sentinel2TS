import numpy as np
from numpy.typing import ArrayLike
import torch
from sentinel2_ts.utils.process_data import get_state


def compute_mode_amplitude_koopman(
    data: ArrayLike,
    model: torch.nn.Module,
    eigenvectors: torch.Tensor,
    x_range: int,
    y_range: int,
) -> ArrayLike:
    """
    Compute the mode amplitude map for the Koopman AE model

    Args:
        data (ArrayLike): Reflectance map
        model (torch.nn.Module): Model
        eigenvectors (torch.Tensor): Eigenvectors from the k matrix in the koopman model
        x_range (int): x range
        y_range (int): y range

    Returns:
        ArrayLike: Mode amplitude map
    """
    mode_amplitude_map = np.zeros((x_range, y_range, 20))

    for x in range(x_range):
        for y in range(y_range):
            mode_amplitude_map[x, y, :] = (
                model.decode(
                    torch.pinverse(eigenvectors)
                    @ model.encode(get_state(data[:, :, x, y], 0))
                )
                .detach()
                .numpy()
            )
    return mode_amplitude_map


def compute_mode_amplitude_map_linear(
    data: ArrayLike,
    eigenvectors: torch.Tensor,
    x_range: int,
    y_range: int,
) -> ArrayLike:
    """
    Compute the mode amplitude map for the linear model

    Args:
        data (ArrayLike): Reflectance map
        eigenvectors (torch.Tensor): Eigenvectors from the k matrix
        x_range (int): x range
        y_range (int): y range

    Returns:
        ArrayLike: Mode amplitude map
    """
    mode_amplitude_map = np.zeros((x_range, y_range, 20))

    for x in range(x_range):
        for y in range(y_range):
            mode_amplitude_map[x, y, :] = (
                np.linalg.inv(eigenvectors)
                @ get_state(data[:, :, x, y], 0).detach().numpy()
            )
    return mode_amplitude_map
