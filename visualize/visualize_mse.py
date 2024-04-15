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

from sentinel2_ts.architectures import Linear, LSTM
from sentinel2_ts.utils.load_model import koopman_model_from_ckpt
from sentinel2_ts.utils.process_data import scale_data, get_state_all_data


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
        "--path_matrix_k",
        type=str,
        default=None,
        help="Path to the Koopman operator",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        nargs="+",
        default=[512, 256, 32],
        help="Latent dimension",
    )

    return parser


def main():
    args = create_parser().parse_args()

    if args.mode == "lstm":
        model = LSTM(20, 256, 20)
    if args.mode == "linear":
        model = Linear(20)
    if args.mode == "koopman_ae":
        model = koopman_model_from_ckpt(
            args.ckpt_path, args.path_matrix_k, "koopman_ae", args.latent_dim
        )
    if args.mode == "koopman_unmixer":
        model = koopman_model_from_ckpt(
            args.ckpt_path, args.path_matrix_k, "koopman_unmixer", args.latent_dim
        )

    model.load_state_dict(torch.load(args.ckpt_path))
    model.to(args.device)
    model.eval()

    data = np.load(args.data_path)
    data = scale_data(data, clipping=args.clipping)

    x_range, y_range = data.shape[2], data.shape[3]
    mse_map = np.zeros((x_range, y_range))
    state_map_time_series = get_state_all_data(data)[242:]
    for x in tqdm(range(x_range)):
        prediction = model(state_map_time_series[0, x, :, :].to(args.device), 100)
        mse = torch.mean(
            (
                prediction
                - state_map_time_series[1:, x, :, :].transpose(0, 1).to(args.device)
            )
            ** 2,
            dim=(1, 2),
        )
        mse_map[x, :] = mse.cpu().detach().numpy()

    plt.imshow(mse_map)
    plt.axis("off")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
