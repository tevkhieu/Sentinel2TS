import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sentinel2_ts.runners.clusterizer import Clusterizer
from sentinel2_ts.utils.process_data import scale_data, get_all_states_at_time
from sentinel2_ts.utils.mode_amplitude_map import (
    compute_mode_amplitude_koopman,
    compute_mode_amplitude_map_linear,
)
from sentinel2_ts.utils.load_model import koopman_model_from_ckpt


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

    return parser


def main():
    args = create_argparser().parse_args()

    data = np.load(args.data_path)
    data = scale_data(data, clipping=args.clipping)
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
    match args.cluster_mode:
        case "kmeans":
            cluster_map_dmd = clusterizer.clusterize_kmeans(
                mode_amplitude_map, nb_clusters=args.nb_clusters
            )
        case "gmm":
            cluster_map_dmd = clusterizer.clusterize_gmm(
                mode_amplitude_map,
                nb_components=args.nb_components,
                covariance_type=args.covariance_type,
            )
        case "dbscan":
            cluster_map_dmd = clusterizer.clusterize_dbscan(mode_amplitude_map)
        case _:
            raise ValueError("Invalid cluster mode")

    fig, ax = plt.subplots(1, 1)
    ax.imshow(cluster_map_dmd)
    ax.set_title("Clustering on data represented in the eigenvector basis")
    plt.show()


def extract_mode_amplitude_map(args, data, x_range, y_range):
    matrix_k = torch.load(args.path_matrix_k)
    matrix_k = matrix_k.cpu().detach()
    model = koopman_model_from_ckpt(
        args.ckpt_path, args.path_matrix_k, args.mode, args.latent_dim
    )
    _, eigenvectors = torch.linalg.eig(matrix_k)
    mode_amplitude_map = compute_mode_amplitude_koopman(
        data, model, eigenvectors, x_range, y_range
    )

    return mode_amplitude_map


if __name__ == "__main__":
    main()
