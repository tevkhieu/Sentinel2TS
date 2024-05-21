import argparse

import torch
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

from sentinel2_ts.data.process_data import scale_data
from sentinel2_ts.utils.visualize import plot_all_spectral_signatures


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/fontainebleau_interpolated.npy",
        help="Path to the data",
    )
    parser.add_argument("--x", type=int, default=250, help="x coordinate")
    parser.add_argument("--y", type=int, default=300, help="y coordinate")
    parser.add_argument(
        "--clipping", type=bool, default=True, help="Clipping the data or not"
    )

    return parser


def compute_dmd(data: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    """
    Compute the DMD of the data at point x, y

    Args:
        data (ArrayLike): Time series data

    Returns:
        Tuple[ArrayLike, ArrayLike]: Phi and Lambda DMD modes and eigenvalues
    """
    x_data = data[:-1, :].T
    x_prime_data = data[1:, :].T

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

    data = scale_data(np.load(args.data_path), clipping=True)[:, :, args.x, args.y]

    Phi, eigenvalues = compute_dmd(data)

    initial_amplitudes = np.linalg.pinv(Phi) @ data[1]

    initial_amplitudes = initial_amplitudes[eigenvalues.imag >= 0]
    Phi = Phi[:, eigenvalues.imag >= 0]
    eigenvalues = eigenvalues[eigenvalues.imag >= 0]

    fig, ax = plt.subplots(nrows=1, ncols=2)
    plt.subplots_adjust(bottom=0.25)  # Adjust bottom to make room for the band_slider
    scatter_plot = ax[0].scatter(
        np.angle(eigenvalues),
        np.abs(eigenvalues),
        c=np.log(np.abs(initial_amplitudes)),
    )
    ax[0].vlines(2 * np.pi / 73, 0.96, 1, colors="red", linestyles="dashed")
    plt.colorbar(scatter_plot)

    plot_all_spectral_signatures(ax[1], Phi, eigenvalues)
    ax.legend()
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Amplitude")
    plt.show()

if __name__ == "__main__":
    main()
