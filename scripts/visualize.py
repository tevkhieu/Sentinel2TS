import os
import argparse
import random as rd
import torch
import numpy as np
import matplotlib.pyplot as plt

from sentinel2_ts.architectures.lstm import LSTM
from sentinel2_ts.architectures.linear import Linear
from sentinel2_ts.utils.process_data import get_state


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Visualize inference of a network")

    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to the network checkpoint")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to the data to visualize")
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

    return parser


def main():
    args = create_arg_parser().parse_args()

    if args.mode == "lstm":
        model = LSTM(20, 256, 20)
    if args.mode == "linear":
        model = Linear(20)

    state_dict = torch.load(args.ckpt_path)
    model.load_state_dict(state_dict)
    model.to(args.device)

    x = rd.randint(0, 99) if args.x is None else args.x
    y = rd.randint(0, 99) if args.y is None else args.x
    band = rd.randint(0, 9) if args.band is None else args.band

    data_path = os.path.join(args.dataset_path, f"{x:03}_{y:03}.npy")
    data = np.load(data_path)

    initial_state = get_state(data, 0).view(1, 1, -1).to(args.device)
    prediction = model(initial_state, args.time_span).squeeze().cpu().detach()
    ground_truth = data[: args.time_span]

    mask = np.ones_like(ground_truth) if args.mask is None else np.load(args.mask)[: args.time_span]
    indices = np.where(mask == 1)[0]

    plt.plot(prediction[:, band], label="Prediction")
    plt.plot(indices, ground_truth[indices, band], label="Groundtruth")
    plt.title(f"band number {band} from pixel {x}, {y} from {os.path.split(args.dataset_path)[-1]}")
    plt.axvline([242], 0, 1, c="black", linestyle="dashed", linewidth=3)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
