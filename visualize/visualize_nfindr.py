import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from pysptools.eea import NFINDR

from sentinel2_ts.dataset.process_data import scale_data
from sentinel2_ts.utils.visualize import plot_single_spectral_signature


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None, help="Path to the data")
    parser.add_argument(
        "--clipping", type=bool, default=True, help="Clipping the data or not"
    )
    parser.add_argument(
        "--num_endmembers", type=int, default=5, help="Number of endmembers to extract"
    )
    parser.add_argument("--time", type=int, default=0, help="Time step to visualize")
    return parser


def main():
    args = create_argparser().parse_args()
    data = np.load(args.data_path)
    data = scale_data(data, clipping=args.clipping)[args.time, :, 37:, 33:].transpose(
        1, 2, 0
    )

    endmember_extractor = NFINDR()
    endmembers = endmember_extractor.extract(data, args.num_endmembers)
    endmember_indices = endmember_extractor.get_idx()
    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow((data[:, :, [2, 1, 0]] * 3).clip(0, 1))
    ax[0].scatter(endmember_indices[0][0], endmember_indices[0][1], c="r")
    plt.subplots_adjust(left=0.1, bottom=0.25)
    plot_single_spectral_signature(ax[1], endmembers[0])
    endmember_slider_ax = plt.axes([0.1, 0.1, 0.65, 0.03])
    endmember_slider = Slider(
        endmember_slider_ax,
        "Endmember",
        0,
        endmembers.shape[0] - 1,
        valinit=0,
        valstep=1,
    )

    def update(val):
        ax[0].clear()
        ax[1].clear()
        ax[0].imshow((data[:, :, [2, 1, 0]] * 3).clip(0, 1))
        endmember_index = int(endmember_slider.val)
        ax[0].scatter(
            endmember_indices[endmember_index][0],
            endmember_indices[endmember_index][1],
            c="r",
        )
        plot_single_spectral_signature(ax[1], endmembers[endmember_index])
        fig.canvas.draw_idle()

    endmember_slider.on_changed(update)

    plt.show()


if __name__ == "__main__":
    main()
