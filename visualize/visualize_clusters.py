import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sentinel2_ts.runners.clusterizer import Clusterizer
from sentinel2_ts.dataset.process_data import scale_data
from sentinel2_ts.utils.mode_amplitude_map import (
    extract_mode_amplitude_map,
    compute_mode_amplitude_map_linear,
)


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
    parser.add_argument("--show_tsne", type=bool, default=False, help="Show t-SNE plot")

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

    if args.show_tsne:
        X_tsne = clusterizer.clusterize_tsne(mode_amplitude_map)
        fig, ax = plt.subplots(1, 1)
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_map_dmd.flatten(), s=1)
        fig.legend()
        plt.show()


if __name__ == "__main__":
    main()
