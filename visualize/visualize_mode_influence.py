import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch

from sentinel2_ts.utils.load_model import koopman_model_from_ckpt
from sentinel2_ts.data.process_data import scale_data, get_state


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_matrix_k",
        default="models/linear/linear_fontainebleau_mixed.pt",
        help="Path to the matrix K file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="linear",
        help="linear | koopman_ae chooses which architecture to use",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to the network checkpoint (only for koopman_ae)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/fontainebleau_interpolated.npy",
        help="Path to the data",
    )
    parser.add_argument(
        "--clipping", type=bool, default=True, help="Clipping the data or not"
    )
    return parser


def main():
    args = create_argparser().parse_args()

    data = np.load(args.data_path)
    data = scale_data(data, clipping=args.clipping)

    match args.mode:
        case "linear":
            matrix_k = torch.load(args.path_matrix_k)["k.weight"].cpu().detach()
            eigenvalues, eigenvectors = torch.linalg.eig(matrix_k)

            scaled_mode_concrete = torch.abs(
                torch.pinverse(eigenvectors)
                @ get_state(data[:, :, 115, 195], 0).to(torch.complex64)
            )
            scaled_mode_forest = torch.abs(
                torch.pinverse(eigenvectors)
                @ get_state(data[:, :, 386, 339], 0).to(torch.complex64)
            )

        case "koopman_ae":
            matrix_k = torch.load(args.path_matrix_k)
            matrix_k = matrix_k.cpu().detach()
            model = koopman_model_from_ckpt(args.ckpt_path, args.path_matrix_k)
            eigenvalues, eigenvectors = torch.linalg.eig(matrix_k)

            scaled_mode_concrete = (
                torch.abs(
                    torch.pinverse(eigenvectors)
                    @ model.encode(get_state(data[:, :, 115, 195], 0)).to(
                        torch.complex64
                    )
                )
                .detach()
                .numpy()
            )
            scaled_mode_forest = (
                torch.abs(
                    torch.pinverse(eigenvectors)
                    @ model.encode(get_state(data[:, :, 386, 339], 0)).to(
                        torch.complex64
                    )
                )
                .detach()
                .numpy()
            )

    # filter by eigenvalues in the upper plane of the complex plane
    eigenvectors = eigenvectors[:, eigenvalues.imag > 0]
    scaled_mode_forest = np.log(scaled_mode_forest[eigenvalues.imag > 0])
    scaled_mode_concrete = np.log(scaled_mode_concrete[eigenvalues.imag > 0])
    eigenvalues = eigenvalues[eigenvalues.imag > 0]

    eigenvalues_magnitude = torch.abs(eigenvalues)
    eigenvalues_argument = torch.angle(eigenvalues)

    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(eigenvalues_argument, eigenvalues_magnitude, c=scaled_mode_concrete)
    ax[0].set_xlabel("Argument of the eigenvalues")
    ax[0].set_ylabel("Magnitude of the eigenvalues")
    ax[0].set_title("Concrete")

    im = ax[1].scatter(
        eigenvalues_argument, eigenvalues_magnitude, c=scaled_mode_forest
    )
    ax[1].set_xlabel("Argument of the eigenvalues")
    ax[1].set_ylabel("Magnitude of the eigenvalues")
    ax[1].set_title("Forest")

    fig.colorbar(im)
    plt.show()


if __name__ == "__main__":
    main()
