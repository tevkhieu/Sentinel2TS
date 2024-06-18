import argparse
from sentinel2_ts.dataset.process_data import scale_data
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from tqdm import tqdm
from sentinel2_ts.runners import DatasetGenerator
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
        "--latent_dim",
        type=int,
        nargs="+",
        default=[512, 256, 8],
        help="Latent dimension of the Koopman Autoencoder",
    )
    parser.add_argument("--alpha", type=float, nargs="+", default=None)
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

    artificial_dataset = DatasetGenerator()
    data = np.load("data/fontainebleau_interpolated.npy")

    concrete_path = "data/endmembers/usgs/ASCIIdata_splib07b_rsSentinel2/ChapterA_ArtificialMaterials/S07SNTL2_Cadmium_orange_0_GDS786_ASDFRa_AREF.txt"
    green_grass_path = "data/endmembers/usgs/ASCIIdata_splib07b_rsSentinel2/ChapterV_Vegetation/S07SNTL2_Grass_dry.4+.6green_AMX27_BECKa_AREF.txt"
    oak_leaf_path = "data/endmembers/usgs/ASCIIdata_splib07b_rsSentinel2/ChapterV_Vegetation/S07SNTL2_Oak_Oak-Leaf-1_fresh_ASDFRa_AREF.txt"
    concrete = artificial_dataset.extract_specter(concrete_path)
    green_grass = artificial_dataset.extract_specter(green_grass_path)
    oak_leaf = artificial_dataset.extract_specter(oak_leaf_path)

    grass_time_series = artificial_dataset.generate_neg_exponential_time_series(
        np.array([green_grass]), np.array([1]), np.array([73])
    )
    grass_sin_time_series = artificial_dataset.generate_sin_time_series(
        np.array([green_grass]), np.array([1]), np.array([73])
    )
    concrete_time_series = artificial_dataset.generate_sin_time_series(
        np.array([concrete]), np.array([0.5]), np.array([730])
    )
    forest_time_series = artificial_dataset.generate_sin_time_series(
        np.array([oak_leaf]), np.array([1]), np.array([73])
    )
    if args.alpha is None:
        alpha = artificial_dataset.estimate_alpha(mode_amplitude_map)
        print(alpha)

    data = np.zeros((343, 10, 500, 500))
    abundance_map = np.zeros((500, 500, 4))
    for i in tqdm(range(500)):
        for j in range(500):
            time_series, abundance = artificial_dataset.generate_entangled_time_series(
                np.array(
                    [
                        grass_time_series,
                        grass_sin_time_series,
                        concrete_time_series,
                        forest_time_series,
                    ]
                ),
                alpha,
            )
            data[:, :, i, j] = time_series
            abundance_map[i, j, :] = abundance.flatten()
            np.save(f"datasets/artificial/{i:03}_{j:03}.npy", time_series)
    np.save("data/artificial_data.npy", data)
    np.save("data/artificial_abundance.npy", abundance_map)

    print(abundance)
    fig, ax = plt.subplots()
    plt.imshow(abundance_map[:, :, 0], vmin=0, vmax=1)
    plt.subplots_adjust(left=0.1, bottom=0.25)
    slider_ax = plt.axes([0.1, 0.1, 0.65, 0.03])
    slider = Slider(slider_ax, "Endmember", 0, 3, valinit=0, valstep=1)

    def update(val):
        ax.clear()
        endmember = int(slider.val)
        ax.imshow(abundance_map[:, :, endmember], vmin=0, vmax=1)
        fig.canvas.draw_idle()

    slider.on_changed(update)
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
