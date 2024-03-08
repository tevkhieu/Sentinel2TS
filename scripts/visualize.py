import os
import argparse
import random as rd

import numpy as np
import matplotlib.pyplot as plt

from sentinel2_ts.runners.lit_lstm import LitLSTM
from sentinel2_ts.utils.process_data import get_state

import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

def create_arg_parser():
    parser = argparse.ArgumentParser(description='Description of your program')

    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to the network checkpoint')
    parser.add_argument('--dataset_path', type=str, default=None, help='Path to the data to visualize')
    parser.add_argument('--time_span', type=int, default=342, help='Number of time steps in the future predicted by the network')
    parser.add_argument('--mode', type=str, default="lstm", help='lstm | linear | koopman_ae chooses which architecture to use')
    parser.add_argument('--x', type=int, default=None, help="x value where to compute prediction")
    parser.add_argument('--y', type=int, default=None, help="y value where to compute prediction")
    parser.add_argument('--band', type=int, default=None, help="band to display on the graph")
    parser.add_argument('--mask', type=str, default=None, help="path to the mask file")

    return parser

def main():
    args = create_arg_parser().parse_args()
    if args.mode == "lstm":
        model = LitLSTM.load_from_checkpoint(args.ckpt_path, time_span=args.time_span)
    
    x = rd.randint(0, 99) if args.x is None else args.x
    y = rd.randint(0, 99) if args.y is None else args.x
    band = rd.randint(0, 9) if args.band is None else args.band

    data_path = os.path.join(args.dataset_path, f"{x:03}_{y:03}.npy")
    data = np.load(data_path)

    initial_state = get_state(data, x, y, 0).view(1, 1, -1).to("cuda:0")
    prediction = model(initial_state).squeeze().cpu().detach()
    ground_truth = data[:args.time_span]

    mask = np.ones_like(ground_truth) if args.mask is None else np.load(args.mask)[:args.time_span]
    indices = np.where(mask==1)[0]

    plt.plot(prediction[:, band], label='Prediction')
    plt.plot(indices, ground_truth[indices, band], label='Groundtruth')
    plt.title(f"band number {band} from pixel {x}, {y} from {os.path.split(args.dataset_path)[-1]}")
    plt.axvline([242], 0, 1, c='black', linestyle='dashed', linewidth=3)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()