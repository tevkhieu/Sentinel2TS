import os
import torch
from torch import Tensor
import numpy as np
from numpy.typing import ArrayLike

def scale_data(data: ArrayLike, clipping: bool = True) -> ArrayLike:
    data /= np.max(data)
    data *= 3
    if clipping:
        data[data > 1] = 1
    return data

def get_state(data, x, y, t):
    reflectance = Tensor(data[t])
    reflectance_diff = Tensor(data[t] - data[t - 1])

    initial_state = torch.cat((reflectance, reflectance_diff))

    return initial_state
