import os
import argparse
import random as rd
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Visualize inference of a network")
    parser.add_argument("--data_path", type=str, default=None, help="Path to the data")
    parser.add_argument(
        "--reconstruction_path",
        type=str,
        default=None,
        help="Path to the reconstruction",
    )
    parser.add_argument(
        "--x", type=int, default=None, help="x value where to compute prediction"
    )
    parser.add_argument(
        "--y", type=int, default=None, help="y value where to compute prediction"
    )
    parser.add_argument("--mask", type=str, default=None, help="path to the mask file")
    return parser


@torch.no_grad()
def main():
    args = create_arg_parser().parse_args()

    data = np.load(args.data_path)
    reconstruction = np.load(args.reconstruction_path)
    x = rd.randint(0, data.shape[2]) if args.x is None else args.x
    y = rd.randint(0, data.shape[3]) if args.y is None else args.y

    prediction = reconstruction[:, :, x, y]
    ground_truth = data[:, :, x, y]
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)  # Adjust bottom to make room for the slider

    slider_ax = plt.axes(
        [0.25, 0.1, 0.65, 0.03], facecolor="lightgoldenrodyellow"
    )  # Define the slider's position and siz
    slider = Slider(
        slider_ax, "Band", 0, 9, valinit=0, valstep=1
    )  # Define the slider itself

    def update(val):
        band = int(slider.val)
        ax.clear()
        ax.plot(prediction[:, band], label="Prediction")
        ax.plot(indices, ground_truth[indices, band], label="Groundtruth")
        ax.axvline([242], 0, 1, c="black", linestyle="dashed", linewidth=3)
        ax.legend()
        fig.canvas.draw_idle()  # Redraw the plot

    mask = (
        np.ones_like(ground_truth)
        if args.mask is None
        else np.load(args.mask)[: args.time_span]
    )
    indices = np.where(mask == 1)[0]
    ax.plot(prediction[:, 0], label="Prediction")
    ax.plot(indices, ground_truth[indices, 0], label="Groundtruth")
    ax.axvline([242], 0, 1, c="black", linestyle="dashed", linewidth=3)

    slider.on_changed(update)
    ax.legend()
    plt.title(f"pixel {x}, {y} from {os.path.split(args.data_path)[-1].split('.')[0]}")
    plt.show()


if __name__ == "__main__":
    main()
