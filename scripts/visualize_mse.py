"""
Description: 
Script for visualizing the mean squared error (MSE) of a model on the map.
There must be a way to make this faster because it's fucking slow.

"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch

from sentinel2_ts.architectures.lstm import LSTM
from sentinel2_ts.architectures.linear import Linear
from sentinel2_ts.architectures.koopman_ae import KoopmanAE
from sentinel2_ts.utils.process_data import scale_data, get_state, get_state_time_series

def create_parser():
    parser = argparse.ArgumentParser(description='Script for visualizing MSE')

    parser.add_argument('--mode', type=str, default=None, help='Mode of operation')
    parser.add_argument('--ckpt_path', type=str,default=None, help='Path to checkpoint file')
    parser.add_argument('--data_path', type=str, default=None,help='Path to data file')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument("--clipping", type=bool, default=True, help="Clipping the data or not")
    return parser

def main():
    args = create_parser().parse_args()

    if args.mode == 'lstm':
        model = LSTM(20, 256, 20)
    if args.mode == 'linear':
        model = Linear(20)
    if args.mode == 'koopman_ae':
        model = KoopmanAE(20, [512, 256, 32], device=args.device)
        if args.koopman_operator_path is not None:
            model.K = torch.load(args.koopman_operator_path)

    model.load_state_dict(torch.load(args.ckpt_path))
    model.to(args.device)
    model.eval()

    data = np.load(args.data_path)
    data = scale_data(data, clipping=args.clipping)

    x_range, y_range, time_span = data.shape[2], data.shape[3], data.shape[0]
    mse_map = np.zeros((x_range, y_range))
    for x in range(x_range):
        for y in range(y_range):
            state = get_state(data[:, :, x, y], 1).to(args.device)
            state_time_series = get_state_time_series(data[:, :, x, y], 1, time_span=time_span-1).to(args.device)
            prediction = model(state, time_span)
            mse = torch.mean((state_time_series[-100:] - prediction[-100:]) ** 2)
            mse_map[x, y] = mse.item()
    
    plt.imshow(mse_map)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()