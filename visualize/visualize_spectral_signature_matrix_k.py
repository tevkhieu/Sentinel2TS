import numpy as np
import argparse
import torch
import matplotlib.pyplot as plt
from sentinel2_ts.architectures.koopman_ae import KoopmanAE
from sentinel2_ts.utils.visualize import plot_all_spectral_signatures


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
    matrix_k = torch.load(args.path_matrix_k)["k.weight"].cpu().detach().numpy()

    eigenvalues, eigenvectors = np.linalg.eig(matrix_k)

    # plot the unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(theta), np.sin(theta))

    # Check if all the eigenvalues are in the unit circle
    plt.plot(eigenvalues.real, eigenvalues.imag, "x", label="Eigenvalues")

    # make the plot orthonormal
    plt.axis("equal")
    plt.legend()
    plt.title("Eigenvalues of the Koopman operator")
    plt.show()

    if args.mode == "koopman_ae":
        eigenvectors = torch.Tensor(eigenvectors)
        model = KoopmanAE(20, [512, 256, 32])
        model.load_state_dict(torch.load(args.ckpt_path))
        eigenvectors = model.decode(eigenvectors).detach().numpy()

    plot_all_spectral_signatures(args, eigenvectors)
    plt.show()


if __name__ == "__main__":
    main()
