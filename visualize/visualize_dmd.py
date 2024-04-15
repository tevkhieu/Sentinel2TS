import argparse
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from tqdm import tqdm
from numpy.typing import ArrayLike

from sentinel2_ts.utils.process_data import scale_data
from sentinel2_ts.utils.visualize import axes_off


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to the data")
    parser.add_argument(
        "--load_pre_computed_data",
        type=bool,
        default=False,
        help="Load pre-computed data or not",
    )

    return parser


def compute_dmd(data: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    """
    Compute the DMD of the data at point x, y

    Args:
        data (ArrayLike): Time series data

    Returns:
        Tuple[ArrayLike, ArrayLike]: eigenvectors and Lambda DMD modes and eigenvalues
    """
    data = data.reshape(data.shape[0], -1)
    X = data[:-1, :]
    Y = data[1:, :]

    U, s, Vh = np.linalg.svd(X.T, full_matrices=False)

    r = len(s)
    S_r = np.diag(s[:r])

    A_tilde = U.T @ Y.T @ Vh.T @ np.linalg.inv(S_r)
    eigenvalues, eigenvectors = np.linalg.eig(A_tilde)

    temp = Y.T @ Vh.T @ np.diag(1 / s)

    modes = (temp @ eigenvectors) / eigenvalues[np.newaxis, :]

    initial_amplitudes = np.linalg.pinv(modes @ np.diag(eigenvalues)) @ data[1].reshape(
        -1
    )

    return (
        eigenvalues,
        modes.reshape(500, 500, 342).transpose(2, 0, 1),
        initial_amplitudes,
    )


def main():
    args = create_argparser().parse_args()

    if args.load_pre_computed_data:
        eigenvalues = np.load("eigenvalues.npy")
        eigenvectors = np.load("eigenvectors.npy")
        initial_amplitudes = np.load("initial_amplitudes.npy")
    else:
        data = scale_data(np.load(args.data_path), clipping=True)

        eigenvalues_array = []
        eigenvectors_array = []
        amplitude_maps_array = []

        for band in tqdm(range(data.shape[1])):
            eigenvalues, eigenvectors, initial_amplitudes = compute_dmd(
                data[:, band, :, :]
            )
            eigenvalues_array.append(eigenvalues)
            eigenvectors_array.append(eigenvectors)
            amplitude_maps_array.append(initial_amplitudes)

        eigenvalues = np.stack(eigenvalues_array)
        eigenvectors = np.stack(eigenvectors_array)
        initial_amplitudes = np.stack(amplitude_maps_array)

        np.save("eigenvalues.npy", eigenvalues)
        np.save("eigenvectors.npy", eigenvectors)
        np.save("initial_amplitudes.npy", initial_amplitudes)

    # TODO Fix this someday lmao
    # for i in range(eigenvalues.shape[0]):
    #     eigenvectors = eigenvectors[i, eigenvalues[i].imag > 0]
    #     initial_amplitudes = initial_amplitudes[i, eigenvalues[i].imag > 0]
    #     eigenvalues = eigenvalues[i, eigenvalues[i].imag > 0]

    fig, ax = plt.subplots(nrows=1, ncols=2)
    plt.subplots_adjust(bottom=0.25)  # Adjust bottom to make room for the band_slider
    scatter_plot = ax[0].scatter(
        np.angle(eigenvalues[0]),
        np.abs(eigenvalues[0]),
        c=np.log(np.abs(initial_amplitudes[0])),
    )
    im = ax[1].imshow(eigenvectors.real[0, 0])

    band_slider_ax = plt.axes(
        [0.25, 0.1, 0.65, 0.03], facecolor="lightgoldenrodyellow"
    )  # Define the band_slider's position and size
    band_slider = Slider(
        band_slider_ax,
        "Band index",
        0,
        eigenvalues.shape[0] - 1,
        valinit=0,
        valstep=1,
    )

    mode_slider_ax = plt.axes(
        [0.25, 0.05, 0.65, 0.03], facecolor="lightgoldenrodyellow"
    )
    mode_slider = Slider(
        mode_slider_ax,
        "Mode index",
        0,
        342 - 1,
        valinit=0,
        valstep=1,
    )

    def update(val):
        band_index = band_slider.val
        mode_index = mode_slider.val
        im.set_data(eigenvectors.real[int(band_index), int(mode_index)])
        scatter_plot.set_offsets(
            np.array(
                [
                    [
                        np.angle(eigenvalues[int(band_index)]),
                        np.abs(eigenvalues[int(band_index)]),
                    ]
                ]
            )
            .squeeze(0)
            .transpose(1, 0)
        )
        scatter_plot.set_array(np.log(np.abs(initial_amplitudes[int(band_index)])))
        fig.canvas.draw_idle()

    band_slider.on_changed(update)
    mode_slider.on_changed(update)

    plt.colorbar(im)
    plt.colorbar(scatter_plot)
    plt.show()


if __name__ == "__main__":
    main()
