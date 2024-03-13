import argparse
import numpy as np
from numpy.typing import ArrayLike

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to the data")
    parser.add_argument("--x", type=int, help="x coordinate")
    parser.add_argument("--y", type=int, help="y coordinate")

    return parser


def compute_dmd(data: ArrayLike, x: int, y:int) -> tuple[ArrayLike, ArrayLike]:
    """
    Compute the DMD of the data at point x, y

    Args:
        data (ArrayLike): Time series data
        x (int): x coordinate
        y (int): y coordinate

    Returns:
        Tuple[ArrayLike, ArrayLike]: Phi and Lambda DMD modes and eigenvalues
    """
    data = data[:, :, x, y]

    U, S, Vh = np.linalg.svd(data[:, :-1], full_matrices=False)

    data_inv = Vh.T @ np.diag(1 / S) @ U.T

    A_tilde = U.T @ data_inv @ Vh.T @ np.diag(1 / S)

    W, L = np.linalg.eig(A_tilde)

    Phi = data_inv @ Vh.T @ np.diag(1 / S) @ W

    return Phi, L


def main():
    args = create_argparser().parse_args()
    data = np.load(args.data_path)



    Phi, L = compute_dmd(data, 0, 0)

    print(L)

    