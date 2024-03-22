"""
Description: 
Script for visualizing the mean squared error (MSE) of a model on the map.
There must be a way to make this faster because it's fucking slow.

"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from sentinel2_ts.architectures.lstm import LSTM
from sentinel2_ts.architectures.linear import Linear
from sentinel2_ts.architectures.koopman_ae import KoopmanAE
from sentinel2_ts.utils.process_data import scale_data, get_state, get_state_time_series


def create_parser():
    parser = argparse.ArgumentParser(description="Script for visualizing MSE")

    parser.add_argument("--mode", type=str, default=None, help="Mode of operation")
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="Path to checkpoint file"
    )
    parser.add_argument("--data_path", type=str, default=None, help="Path to data file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument(
        "--clipping", type=bool, default=True, help="Clipping the data or not"
    )
    parser.add_argument(
        "--koopman_operator_path",
        type=str,
        default=None,
        help="Path to the Koopman operator",
    )
    return parser


def main():
    args = create_parser().parse_args()

    if args.mode == "lstm":
        model = LSTM(20, 256, 20)
    if args.mode == "linear":
        model = Linear(20)
    if args.mode == "koopman_ae":
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
    for x in tqdm(range(x_range)):
        reflectance = torch.Tensor(data[1, :, x, :].transpose(1, 0))
        reflectance_diff = torch.Tensor(
            data[1, :, x, :].transpose(1, 0) - data[0, :, x, :].transpose(1, 0)
        )
        state = (
            torch.cat((reflectance, reflectance_diff), dim=1)
            .unsqueeze(1)
            .to(args.device)
        )
        state_time_series = torch.cat(
            (
                torch.Tensor(data[1:, :, x, :].transpose(2, 0, 1)),
                torch.Tensor(
                    data[1:, :, x, :].transpose(2, 0, 1)
                    - data[:-1, :, x, :].transpose(2, 0, 1)
                ),
            ),
            dim=2,
        ).to(args.device)
        prediction = model(state, time_span)
        mse = torch.mean((prediction - state_time_series) ** 2, dim=(1, 2))
        mse_map[x, :] = mse.cpu().detach().numpy()

    plt.imshow(mse_map)
    plt.axis("off")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
