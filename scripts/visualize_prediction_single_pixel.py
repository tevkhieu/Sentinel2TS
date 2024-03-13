import os
import argparse
import random as rd
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from sentinel2_ts.architectures.lstm import LSTM
from sentinel2_ts.architectures.linear import Linear
from sentinel2_ts.architectures.koopman_ae import KoopmanAE
from sentinel2_ts.utils.process_data import get_state_from_data, scale_data


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Visualize inference of a network")

    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to the network checkpoint")
    parser.add_argument("--data_path", type=str, default=None, help="Path to the data to visualize")
    parser.add_argument(
        "--time_span", type=int, default=342, help="Number of time steps in the future predicted by the network"
    )
    parser.add_argument(
        "--mode", type=str, default="lstm", help="lstm | linear | koopman_ae chooses which architecture to use"
    )
    parser.add_argument("--x", type=int, default=None, help="x value where to compute prediction")
    parser.add_argument("--y", type=int, default=None, help="y value where to compute prediction")
    parser.add_argument("--band", type=int, default=None, help="band to display on the graph")
    parser.add_argument("--mask", type=str, default=None, help="path to the mask file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device used")
    parser.add_argument("--koopman_operator_path", type=str, default=None, help="Path to the matrix K (only for koopman_ae)")
    

    return parser


def main():
    args = create_arg_parser().parse_args()

    if args.mode == "lstm":
        model = LSTM(20, 256, 20)
    if args.mode == "linear":
        model = Linear(20)
    if args.mode == "koopman_ae":
        model = KoopmanAE(20, [512, 256, 32], device=args.device)
        if args.koopman_operator_path is not None:
            model.K = torch.load(args.koopman_operator_path)

    state_dict = torch.load(args.ckpt_path)
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    data = np.load(args.data_path)
    data = scale_data(data, clipping=False)

    x = rd.randint(0, data.shape[2]) if args.x is None else args.x
    y = rd.randint(0, data.shape[3]) if args.y is None else args.x
    band = rd.randint(0, 9) if args.band is None else args.band

    initial_state = get_state_from_data(data, x, y, 0).view(1, 1, -1).to(args.device)
    prediction = model(initial_state, args.time_span).squeeze().cpu().detach()
    ground_truth = data[: args.time_span, :, x, y]

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)  # Adjust bottom to make room for the slider

    slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor="lightgoldenrodyellow")  # Define the slider's position and siz
    slider = Slider(slider_ax, "Band", 0, 9, valinit=0, valstep=1)  # Define the slider itself

    def update(val):
        band = int(slider.val)
        ax.clear()
        ax.plot(prediction[:, band], label="Prediction")
        ax.plot(indices, ground_truth[indices, band], label="Groundtruth")
        ax.axvline([242], 0, 1, c="black", linestyle="dashed", linewidth=3)
        fig.canvas.draw_idle()  # Redraw the plot

    mask = np.ones_like(ground_truth) if args.mask is None else np.load(args.mask)[: args.time_span]
    indices = np.where(mask == 1)[0]
    ax.plot(prediction[:, band], label="Prediction")
    ax.plot(indices, ground_truth[indices, band], label="Groundtruth")
    ax.axvline([242], 0, 1, c="black", linestyle="dashed", linewidth=3)
    
    slider.on_changed(update)
    plt.legend()    
    plt.title(f"pixel {x}, {y} from {os.path.split(args.data_path)[-1].split('.')[0]}")
    plt.show()


if __name__ == "__main__":
    main()
