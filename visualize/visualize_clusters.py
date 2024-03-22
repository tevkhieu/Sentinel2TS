import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sentinel2_ts.runners.clusterizer import Clusterizer
from sentinel2_ts.utils.process_data import scale_data, get_state_all_data
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

    return parser


def main():
    args = create_argparser().parse_args()

    data = np.load(args.data_path)
    data = scale_data(data, clipping=args.clipping)
    x_range, y_range = data.shape[2], data.shape[3]

    if args.mode == "linear":
        matrix_k = torch.load(args.path_matrix_k)["k.weight"].cpu().detach().numpy()
        eigenvalues, eigenvectors = np.linalg.eig(matrix_k)
        mode_amplitude_map = compute_mode_amplitude_map_linear(
            data, eigenvectors, x_range, y_range
        )

    elif args.mode == "koopman_ae":
        matrix_k = torch.load(args.path_matrix_k)
        matrix_k = matrix_k.cpu().detach().numpy()
        model = koopman_model_from_ckpt(args.ckpt_path, args.path_matrix_k)
        eigenvalues, eigenvectors = np.linalg.eig(matrix_k)
        eigenvectors = torch.Tensor(eigenvectors)
        mode_amplitude_map = compute_mode_amplitude_koopman(
            data, model, eigenvectors, x_range, y_range
        )

    clusterizer = Clusterizer()
    cluster_map_dmd = clusterizer.clusterize_kmeans(
        mode_amplitude_map, nb_clusters=args.nb_clusters
    )
    cluster_map_baseline = clusterizer.clusterize_kmeans(
        get_state_all_data(data)[0], nb_clusters=args.nb_clusters
    )

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(cluster_map_dmd)
    ax[0].set_title("Clustering on data represented in the eigenvector basis")
    ax[1].imshow(cluster_map_baseline)
    ax[1].set_title("Clustering on the raw data")
    for axis in ax:
        axis.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
