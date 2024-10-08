import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sentinel2_ts.runners.clusterizer import Clusterizer
from sentinel2_ts.dataset.process_data import scale_data, get_all_states_at_time
from sentinel2_ts.utils.mode_amplitude_map import (
    compute_mode_amplitude_koopman,
    compute_mode_amplitude_map_linear,
)
from sentinel2_ts.utils.load_model import koopman_model_from_ckpt, load_data


def create_argparser():
    parser = argparse.ArgumentParser(description="Visualize Clusters")
    parser.add_argument("--data_path", type=str, help="Path to the data file")
    parser.add_argument("--ckpt_path", type=str, help="Path to the checkpoint file")
    parser.add_argument("--mode", type=str, help="linear | koopman_ae")
    parser.add_argument(
        "--clipping", type=bool, default=True, help="Clipping the data or not"
    )
    parser.add_argument("--path_matrix_k", type=str, help="Path to the matrix K")
    parser.add_argument(
        "--nb_clusters",
        type=int,
        default=None,
        help="Number of clusters in KMeans clustering",
    )
    parser.add_argument(
        "--cluster_mode", type=str, default="kmeans", help="kmeans | gmm | dbscan"
    )
    parser.add_argument(
        "--nb_components",
        type=int,
        default=None,
        help="Number of components in GMM clustering",
    )
    parser.add_argument(
        "--covariance_type",
        type=str,
        default=None,
        help="Covariance type in GMM clustering",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        nargs="+",
        default=[512, 256, 8],
        help="Latent dimension of the Koopman Autoencoder",
    )
    parser.add_argument("--scale_data", type=bool, help="Scale data")
    parser.add_argument("--size", type=int, default=20, help="Size of the data")
    return parser


def main():
    args = create_argparser().parse_args()

    data = load_data(args)
    x_range, y_range = data.shape[2], data.shape[3]

    match args.mode:
        case "linear":
            matrix_k = torch.load(args.path_matrix_k)["k.weight"].cpu().detach().numpy()
            eigenvalues, eigenvectors = np.linalg.eig(matrix_k)
            mode_amplitude_map = compute_mode_amplitude_map_linear(
                data, eigenvectors, x_range, y_range
            )
        case "koopman_ae":
            mode_amplitude_map = extract_mode_amplitude_map(
                args, data, x_range, y_range
            )
        case "koopman_unmixer":
            mode_amplitude_map = extract_mode_amplitude_map(
                args, data, x_range, y_range
            )

    clusterizer = Clusterizer()
    proba_map = clusterizer.proba_gmm(
        mode_amplitude_map,
        nb_components=args.nb_components,
        covariance_type=args.covariance_type,
    )

    fig, ax = plt.subplots(1, args.nb_components)
    plt.subplots_adjust(bottom=0.25)  # Adjust bottom to make room for the slider
    for i in range(args.nb_components):
        ax[i].set_title(f"Component {i}")
        ax[i].axis("off")
        im = ax[i].imshow(proba_map[:, :, i])
    plt.colorbar(im)
    plt.show()


def extract_mode_amplitude_map(args, data, x_range, y_range):
    matrix_k = torch.load(args.path_matrix_k)
    matrix_k = matrix_k.cpu().detach()
    model = koopman_model_from_ckpt(
        args.size, args.ckpt_path, args.path_matrix_k, args.mode, args.latent_dim
    )
    _, eigenvectors = torch.linalg.eig(matrix_k)
    mode_amplitude_map = compute_mode_amplitude_koopman(
        data, model, eigenvectors, x_range, y_range
    )

    return mode_amplitude_map


if __name__ == "__main__":
    main()
