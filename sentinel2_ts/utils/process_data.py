import torch
from torch import Tensor
import numpy as np
from numpy.typing import ArrayLike


def scale_data(data: ArrayLike, clipping: bool = True) -> ArrayLike:
    """
    Scale the data

    Args:
        data (ArrayLike): Time series of hyperspectral images
        clipping (bool, optional): Whether to clip the data or not. Defaults to True.

    Returns:
        data (ArrayLike): Scaled and optionaly clipped time series of hyperspectral images
    """
    data /= np.max(data)
    data *= 3
    if clipping:
        data[data > 1] = 1
    return data


def get_state(pixel_data, t):
    """
    Get the state of a pixel a certain time point
    Args:
        pixel_data (ArrayLike): pixel data point
        t (int): time point

    Returns:
        state (Tensor): state containing reflectance and difference of reflectance at time point t
    """
    reflectance = Tensor(pixel_data[t])
    reflectance_diff = Tensor(pixel_data[t] - pixel_data[t - 1])

    state = torch.cat((reflectance, reflectance_diff))

    return state


def get_state_time_series(pixel_data: ArrayLike, initial_time: int, time_span: int):
    """
    Get the time series of states following an initial time

    Args:
        pixel_data (ArrayLike): pixel data point
        initial_time (int): initial time from which the time series is computed

    Returns:
        observed_state_time_series (ArrayLike): time series of observed states
    """
    end_time = initial_time + time_span
    reflectance_time_series = Tensor(pixel_data[initial_time:end_time])
    reflectance_diff_time_series = Tensor(
        pixel_data[initial_time:end_time] - pixel_data[initial_time - 1 : end_time - 1]
    )

    observed_state_time_series = torch.cat((reflectance_time_series, reflectance_diff_time_series), dim=1)
    return observed_state_time_series
