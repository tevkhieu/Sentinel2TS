import argparse
import numpy as np
from numpy.typing import ArrayLike

from sentinel2_ts.utils.visualize import visualize_spectral_signature


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to the data")
    parser.add_argument("--x", type=int, help="x coordinate")
    parser.add_argument("--y", type=int, help="y coordinate")
    parser.add_argument(
        "--rank_approximation",
        type=int,
        default=5,
        help="Number of eigenvectors to plot",
    )

    return parser


def compute_dmd(data: ArrayLike, x: int, y: int) -> tuple[ArrayLike, ArrayLike]:
    """
    Compute the DMD of the data at point x, y

    Args:
        data (ArrayLike): Time series data
        x (int): x coordinate
        y (int): y coordinate

    Returns:
        Tuple[ArrayLike, ArrayLike]: Phi and Lambda DMD modes and eigenvalues
    """
    x_data = data[:-1, :, x, y].T
    x_prime_data = data[1:, :, x, y].T

    # Performing Singular Value Decomposition (SVD)
    U, S, Vh = np.linalg.svd(x_data, full_matrices=False)

    # Constructing the approximation of the A matrix
    A_tilde = U.T @ x_prime_data @ Vh.T @ np.linalg.inv(np.diag(S))

    # Calculating eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A_tilde)

    # Constructing the DMD modes Phi
    Phi = x_prime_data @ Vh.T @ np.linalg.inv(np.diag(S)) @ eigenvectors

    return Phi, eigenvalues


def main():
    args = create_argparser().parse_args()
    data = np.load(args.data_path)

    Phi, eigenvalues = compute_dmd(data, args.x, args.y)

    print(f"Eigenvalues: {eigenvalues}")

    visualize_spectral_signature(args, Phi)


if __name__ == "__main__":
    main()
