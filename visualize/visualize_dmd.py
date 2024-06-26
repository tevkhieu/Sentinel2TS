import argparse
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from tqdm import tqdm
from numpy.typing import ArrayLike

from sentinel2_ts.utils.load_model import load_data
from sentinel2_ts.runners import DynamicalModeDecomposition


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to the data")
    parser.add_argument(
        "--load_pre_computed_data",
        type=bool,
        default=False,
        help="Load pre-computed data or not",
    )
    parser.add_argument(
        "--scale_data", type=bool, default=True, help="Scale data or not"
    )
    parser.add_argument(
        "--clipping", type=bool, default=True, help="Clipping the data or not"
    )
    return parser


def main():
    args = create_argparser().parse_args()

    if args.load_pre_computed_data:
        eigenvalues = np.load("eigenvalues.npy")
        eigenvectors = np.load("eigenvectors.npy")
        initial_amplitudes = np.load("initial_amplitudes.npy")
    else:
        data = load_data(args)

        dmd = DynamicalModeDecomposition()
        eigenvalues, eigenvectors, initial_amplitudes = dmd.compute_dmd(data)
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
    im = ax[1].imshow(eigenvectors.real[0])

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
        mode_index = mode_slider.val
        im.set_data(eigenvectors.real[int(mode_index)])
        fig.canvas.draw_idle()

    mode_slider.on_changed(update)

    plt.colorbar(im)
    plt.colorbar(scatter_plot)
    plt.show()


if __name__ == "__main__":
    main()
