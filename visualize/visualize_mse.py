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

from sentinel2_ts.architectures import (
    Linear,
    LSTM,
    Disentangler,
    AbundanceDisentangler,
    SpectralDisentangler,
)
from sentinel2_ts.utils.load_model import koopman_model_from_ckpt
from sentinel2_ts.dataset.process_data import scale_data, get_state_all_data


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


@torch.no_grad()
def main():
    args = create_parser().parse_args()
    match args.mode:
        case "lstm":
            model = LSTM(20, 256, 20)
            model.load_state_dict(torch.load(args.ckpt_path))
        case "linear":
            model = Linear(20)
            model.load_state_dict(torch.load(args.ckpt_path))
        case "koopman_ae":
            model = koopman_model_from_ckpt(
                args.ckpt_path, args.path_matrix_k, "koopman_ae", args.latent_dim
            )
        case "koopman_unmixer":
            model = koopman_model_from_ckpt(
                args.ckpt_path, args.path_matrix_k, "koopman_unmixer", args.latent_dim
            )
        case "disentangler":
            model = Disentangler(size=20, latent_dim=64, num_classes=8)
            state_dict_spectral_disentangler = torch.load(
                "models/disentangler/best_spectral_disentangler.pt"
            )
            model.load_state_dict(state_dict_spectral_disentangler)
    model.to(args.device)
    model.eval()

    data = np.load(args.data_path)
    data = scale_data(data, clipping=args.clipping)

    x_range, y_range = data.shape[2], data.shape[3]
    mse_map = np.zeros((x_range, y_range))
    if not args.mode == "disentangler":
        state_map_time_series = get_state_all_data(data)[242:]
    else:
        state_map = (
            get_state_all_data(data).transpose(0, -1).transpose(0, -2).transpose(0, -3)
        )

    total_mse = 0
    for x in tqdm(range(x_range)):
        if args.mode == "disentangler":
            prediction = model(state_map[x, :, :].to(args.device))
            squarred_error = (
                prediction.transpose(1, 2).transpose(0, 1)[242:, :, :10]
                - state_map[x, :, :10, 242:]
                .transpose(1, 2)
                .transpose(0, 1)
                .to(args.device)
            ) ** 2

        else:
            prediction = model(state_map_time_series[0, x, :, :].to(args.device), 100)
            squarred_error = (
                prediction[:, :, :10]
                - state_map_time_series[1:, x, :, :10].transpose(0, 1).to(args.device)
            ) ** 2

        total_mse += squarred_error.sum()
        mse = torch.mean(squarred_error, dim=(0, 2))
        mse_map[x, :] = mse.cpu().detach().numpy()

    print(f"Total MSE: {1e3 * total_mse/(x_range*y_range*100*10) :.4f}")

    plt.imshow(mse_map)
    plt.axis("off")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
