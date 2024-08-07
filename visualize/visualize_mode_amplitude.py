import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
from sklearn.decomposition import PCA
from matplotlib.widgets import Slider

from sentinel2_ts.utils.load_model import koopman_model_from_ckpt
from sentinel2_ts.dataset.process_data import scale_data
from sentinel2_ts.utils.mode_amplitude_map import (
    compute_mode_amplitude_koopman,
    compute_mode_amplitude_map_linear,
)
from sentinel2_ts.utils.visualize import axes_off


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
    parser.add_argument("--data_path", type=str, default=None, help="Path to the data")
    parser.add_argument(
        "--clipping", type=bool, default=True, help="Clipping the data or not"
    )
    return parser


def main():
    args = create_argparser().parse_args()

    data = np.load(args.data_path)
    data = scale_data(data, clipping=args.clipping)
    x_range, y_range = data.shape[2], data.shape[3]

    match args.mode:
        case "linear":
            matrix_k = torch.load(args.path_matrix_k)["k.weight"].cpu().detach()
            eigenvalues, eigenvectors = torch.linalg.eig(matrix_k)
            mode_amplitude_map = compute_mode_amplitude_map_linear(
                data, eigenvectors, x_range, y_range
            )
        case "koopman_ae":
            matrix_k = torch.load(args.path_matrix_k)
            matrix_k = matrix_k.cpu().detach()
            model = koopman_model_from_ckpt(args.ckpt_path, args.path_matrix_k)
            eigenvalues, eigenvectors = torch.linalg.eig(matrix_k)
            mode_amplitude_map = compute_mode_amplitude_koopman(
                data, model, eigenvectors, x_range, y_range
            )

    filtered_eigenvalues = eigenvalues[eigenvalues.imag > 0]
    mode_amplitude_map = mode_amplitude_map[..., eigenvalues.imag > 0]

    # Compute PCA to visualize the modes on the data
    pca = PCA(n_components=3)
    data_pca = pca.fit_transform(
        data[0, :, :, :].transpose(1, 2, 0).reshape(-1, 10)
    ).reshape(500, 500, 3)

    fig, ax = plt.subplots(nrows=1, ncols=3)
    plt.subplots_adjust(bottom=0.25)  # Adjust bottom to make room for the slider
    ax[0].imshow((data[0, [2, 1, 0], :, :] * 3).clip(0, 1).transpose(1, 2, 0))
    im_pca = ax[1].imshow(data_pca)
    im = ax[2].imshow(mode_amplitude_map[..., 0])
    slider_ax = plt.axes(
        [0.25, 0.1, 0.65, 0.03], facecolor="lightgoldenrodyellow"
    )  # Define the slider's position and size
    slider = Slider(
        slider_ax,
        "Band index",
        0,
        filtered_eigenvalues.shape[0] - 1,
        valinit=0,
        valstep=1,
    )  # Define the slider itself

    def update(val):
        band_index = slider.val
        im.set_data(mode_amplitude_map[..., int(band_index)])
        ax[2].set_title(
            f"Mode amplitude map {5 * 2  * np.pi / np.angle(filtered_eigenvalues[int(band_index)]): .4f} days"
        )
        fig.canvas.draw_idle()  # Redraw the plot

    slider.on_changed(update)  # Call update when the slider value is changed
    ax[0].set_title("RGB image")
    ax[1].set_title("PCA")
    ax[2].set_title(
        f"Mode log-amplitude map {5 * 2 * np.pi /np.angle(filtered_eigenvalues[0]): .4f} days"
    )
    plt.colorbar(im)
    axes_off(ax)
    plt.show()


if __name__ == "__main__":
    main()
