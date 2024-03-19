import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
from sklearn.decomposition import PCA
from matplotlib.widgets import Slider

from sentinel2_ts.architectures.koopman_ae import KoopmanAE
from sentinel2_ts.architectures.linear import Linear
from sentinel2_ts.utils.process_data import scale_data, get_state


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

    matrix_k = torch.load(args.path_matrix_k)["k.weight"].cpu().detach().numpy()

    eigenvalues, eigenvectors = np.linalg.eig(matrix_k)

    if args.mode == "koopman_ae":
        eigenvectors = torch.Tensor(eigenvectors)
        model = KoopmanAE(20, [512, 256, 32])
        model.load_state_dict(torch.load(args.ckpt_path))
        eigenvectors = model.decode(eigenvectors).detach().numpy()

    # Compute PCA to visualize the modes on the data
    pca = PCA(n_components=3)
    pca.fit(data[0, :, :, :].transpose(1, 2, 0).reshape(-1, 10))
    data_pca = pca.transform(data[0, :, :, :].transpose(1, 2, 0).reshape(-1, 10)).reshape(500, 500, 3)

    x_range, y_range, band_range = data.shape[2], data.shape[3], data.shape[1]
    mode_amplitude_map = np.zeros((x_range, y_range, 20))

    for x in range(x_range):
        for y in range(y_range):
            mode_amplitude_map[x, y, :] = (
                np.linalg.inv(eigenvectors)
                @ get_state(data[:, :, x, y], 0).detach().numpy()
            )


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
        19,
        valinit=0,
        valstep=1,
    )  # Define the slider itself

    def update(val):
        band_index = slider.val
        im.set_data(mode_amplitude_map[..., int(band_index)])
        fig.canvas.draw_idle()  # Redraw the plot

    slider.on_changed(update)  # Call update when the slider value is changed
    ax[0].set_title("RGB image")
    ax[1].set_title("PCA")
    ax[2].set_title("Mode amplitude map")
    # plt.colorbar(im)
    axes_off(ax)
    plt.show()

def axes_off(ax):
    for a in ax:
        a.axis("off")


if __name__ == "__main__":
    main()
