import os
import argparse
import random as rd
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from sentinel2_ts.utils.load_model import load_model, load_data
from sentinel2_ts.dataset.process_data import (
    get_state_from_data,
    scale_data,
    get_state_time_series,
)

from sentinel2_ts.utils.visualize import plot_single_spectral_signature


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Visualize inference of a network")

    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="Path to the network checkpoint"
    )
    parser.add_argument(
        "--data_path", type=str, default=None, help="Path to the data to visualize"
    )
    parser.add_argument("--clipping", type=bool, help="Clipping the data or not")
    parser.add_argument(
        "--time_span",
        type=int,
        default=343,
        help="Number of time steps in the future predicted by the network",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="lstm",
        help="lstm | linear | koopman_ae | koopman_unmixer chooses which architecture to use",
    )
    parser.add_argument(
        "--x", type=int, default=None, help="x value where to compute prediction"
    )
    parser.add_argument(
        "--y", type=int, default=None, help="y value where to compute prediction"
    )
    parser.add_argument(
        "--band", type=int, default=None, help="band to display on the graph"
    )
    parser.add_argument("--mask", type=str, default=None, help="path to the mask file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device used")
    parser.add_argument(
        "--path_matrix_k",
        type=str,
        default=None,
        help="Path to the matrix K (only for koopman_ae)",
    )
    parser.add_argument("--latent_dim", type=int, nargs="+", default=[512, 256, 32])
    parser.add_argument("--scale_data", type=bool, help="Scale data")
    parser.add_argument(
        "--endmembers", type=str, default=None, help="Path to endmembers"
    )
    return parser


@torch.no_grad()
def main():
    args = create_arg_parser().parse_args()

    model = load_model(args)
    state_dict = torch.load(args.ckpt_path)
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    if args.endmembers is not None:
        endmembers = np.load(args.endmembers)
        endmembers = torch.from_numpy(endmembers).to(args.device)

    data = load_data(args)

    x = rd.randint(0, data.shape[2]) if args.x is None else args.x
    y = rd.randint(0, data.shape[3]) if args.y is None else args.y
    band = rd.randint(0, 9) if args.band is None else args.band

    if not args.mode == "disentangler":
        initial_state = get_state_from_data(data, x, y, 1).unsqueeze(0).to(args.device)
        if args.endmembers is not None:
            prediction = (
                model.forward_with_endmembers(initial_state, endmembers)
                .squeeze()
                .cpu()
                .detach()
            )
        else:
            prediction = model(initial_state, args.time_span).squeeze().cpu().detach()

    else:
        initial_state = (
            get_state_time_series(data[:, :, x, y], 1, 342)
            .unsqueeze(0)
            .transpose(1, 2)
            .to(args.device)
        )
        prediction = (
            model(initial_state).squeeze().cpu().detach().transpose(0, 1).numpy()
        )

    ground_truth = data[1 : args.time_span + 1, :, x, y]
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
    ax.plot(prediction[:, band], label="Prediction")
    ax.plot(indices, ground_truth[indices, band], label="Groundtruth")
    ax.axvline([242], 0, 1, c="black", linestyle="dashed", linewidth=3)

    slider.on_changed(update)
    ax.legend()
    plt.title(
        f"{args.mode}, pixel {x}, {y} from {os.path.split(args.data_path)[-1].split('.')[0]}"
    )
    plt.show()


if __name__ == "__main__":
    main()
