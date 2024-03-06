import torch
import numpy as np
from numpy.typing import ArrayLike

def scale_data(data: ArrayLike, clipping: bool = True) -> ArrayLike:
    data /= np.max(data)
    data *= 3
    if clipping:
        data[data > 1] = 1
    return data