import argparse

import torch
import numpy as np
from numpy.typing import ArrayLike

from sentinel2_ts.architectures.koopman_ae import KoopmanAE
from sentinel2_ts.architectures.linear import Linear
from sentinel2_ts.utils.visualize import visualize_spectral_signature
from sentinel2_ts.utils.process_data import get_state_time_series, scale_data


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
    parser.add_argument("--ckpt_path", type=str, help="Path to the network checkpoint")
    parser.add_argument("--mode", type=str, default=None, help="linear | koopman_ae")
    parser.add_argument(
        "--clipping", type=bool, default=True, help="Clipping the data or not"
    )
    parser.add_argument("--path_matrix_k", help="Path to the matrix K file")

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
    print(f"Eigenvalues of DMD on data: {eigenvalues}")
    visualize_spectral_signature(args, Phi)

    if args.mode == "koopman_ae":
        model = KoopmanAE(20, [512, 256, 32])
        model.load_state_dict(torch.load(args.ckpt_path))
        model.K = torch.load(args.path_matrix_k)
    elif args.mode == "linear":
        model = Linear(20)
        model.load_state_dict(torch.load(args.ckpt_path))

    latent_data = model.encode(get_state_time_series(data, 1, 342)).detach().numpy()
    latent_phi, _ = compute_dmd(latent_data)
    decoded_latent_phi = model.decode(torch.Tensor(latent_phi.T)).detach().numpy()

    visualize_spectral_signature(args, decoded_latent_phi)


if __name__ == "__main__":
    main()
