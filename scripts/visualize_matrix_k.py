import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch

from sentinel2_ts.architectures.koopman_ae import KoopmanAE
from sentinel2_ts.utils.visualize import visualize_spectral_signature


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_matrix_k", help="Path to the matrix K file")
    parser.add_argument(
        "--rank_approximation",
        type=int,
        default=5,
        help="Rank approximation of the Koopman operator",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="linear | koopman_ae chooses which architecture to use",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to the network checkpoint (only for koopman_ae)",
    )

    return parser


def is_unit_circle(eigenvalues):
    return all(np.isclose(np.abs(eigenvalues), 1.0, atol=1e-2))


def main():
    args = create_argparser().parse_args()
    matrix_k = torch.load(args.path_matrix_k).cpu().detach().numpy()

    eigenvalues, eigenvectors = np.linalg.eig(matrix_k)

    # Check if all the eigenvalues are in the unit circle
    print(
        f"All the eigenvalues are close to the unit circle with a tolerance of 1e-2: {is_unit_circle(eigenvalues)}"
    )

    if args.mode == "koopman_ae":
        eigenvectors = torch.Tensor(eigenvectors)
        model = KoopmanAE(20, [512, 256, 32])
        model.load_state_dict(torch.load(args.ckpt_path))
        eigenvectors = model.decode(eigenvectors).detach().numpy()

    visualize_spectral_signature(args, eigenvectors)


if __name__ == "__main__":
    main()
