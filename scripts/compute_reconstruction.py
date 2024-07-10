"""
Description: 
Script for visualizing the mean squared error (MSE) of a model on the map.
There must be a way to make this faster because it's fucking slow.
"""

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from sentinel2_ts.utils.load_model import load_model, load_data
from sentinel2_ts.dataset.process_data import get_state_all_data


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
    parser.add_argument("--scale_data", type=bool, help="Scale data")
    parser.add_argument(
        "--endmembers", type=str, default=None, help="Path to endmembers"
    )
    parser.add_argument(
        "--save_folder", type=str, default=None, help="Folder for results"
    )
    parser.add_argument("--size", type=int, default=20, help="Size of the data")
    return parser


@torch.no_grad()
def main():
    args = create_parser().parse_args()
    if args.mode == "cpd":
        weights_path = os.path.join(args.ckpt_path, "weights.npy")
        time_path = os.path.join(args.ckpt_path, "time.npy")
        specters_path = os.path.join(args.ckpt_path, "specters.npy")
        abundance_map_path = os.path.join(args.ckpt_path, "abundance_map.npy")


        weights = np.load(weights_path)
        time = np.load(time_path)
        specters = np.load(specters_path)
        abundance_map = np.load(abundance_map_path)

        x_range, y_range, nb_endmembers = abundance_map.shape

        prediction_map = np.zeros((time.shape[0], specters.shape[0], abundance_map.shape[0], abundance_map.shape[1]))
        for i in range(nb_endmembers):
            prediction_map += weights[i] * time[:, i].reshape(-1, 1, 1, 1) * abundance_map[:, :, i].reshape(1, 1, x_range, y_range) * specters[:, i].reshape(1, specters.shape[0], 1, 1)


    elif args.mode == "dynamical_unmixing":
        abundance_map_path = os.path.join(args.ckpt_path, "abundance_map.npy")
        specter_path = os.path.join(args.ckpt_path, "specter.npy")

        abundance_map = np.load(abundance_map_path)
        specter = np.load(specter_path)

        x_range, y_range, nb_endmembers = abundance_map.shape

        prediction_map = np.zeros((specter.shape[:2] + abundance_map.shape[:2]))
        for i in range(nb_endmembers):
            prediction_map += abundance_map[:, :, i].reshape(1, 1, x_range, y_range) * specter[:, :, i].reshape(specter.shape[0], specter.shape[1], 1, 1)

    else:
        model = load_model(args)
        model.to(args.device)
        model.eval()

        if args.endmembers is not None:
            endmembers = np.load(args.endmembers)
            endmembers = torch.from_numpy(endmembers).to(args.device)

        data = data = load_data(args)

        time_range, nb_band, x_range, y_range = data.shape
        prediction_map = np.zeros_like(data)
        prediction_map[0] = data[0]
        if not args.mode == "disentangler":
            state_map_time_series = get_state_all_data(data)
        else:
            state_map = (
                get_state_all_data(data).transpose(0, -1).transpose(0, -2).transpose(0, -3)
            )

        for x in tqdm(range(x_range)):
            if args.mode == "disentangler":
                if args.endmembers is not None:
                    prediction = model.forward_with_endmembers(
                        state_map[x, :, :].to(args.device), endmembers
                    )
                    prediction = prediction.transpose(1, 2)[:, :nb_band, :].transpose(0, 2)
                else:
                    prediction = model(state_map[x, :, :].to(args.device))
                    prediction = prediction.transpose(0, 2)[:, :nb_band, :]
            else:
                prediction = model(
                    state_map_time_series[0, x, :, :].to(args.device), time_range
                )
                prediction = prediction.transpose(0, 1).transpose(1, 2)[:, :nb_band, :]

            prediction_map[1:, :, x, :] = prediction.cpu().numpy()

    os.makedirs(args.save_folder, exist_ok=True)
    np.save(os.path.join(args.save_folder, "reconstruction.npy"), prediction_map)


if __name__ == "__main__":
    main()
