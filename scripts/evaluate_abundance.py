import argparse
import os
from itertools import permutations
import numpy as np
import torch
import tensorly as tl
from tensorly.decomposition import CP_NN_HALS
from tqdm import tqdm
import matplotlib.pyplot as plt

from sentinel2_ts.runners import Clusterizer, DynamicalModeDecomposition
from sentinel2_ts.utils.load_model import load_model, load_data
from sentinel2_ts.utils.process_data import get_state_all_data
from sentinel2_ts.utils.mode_amplitude_map import extract_mode_amplitude_map


def create_parser():
    parser = argparse.ArgumentParser(description="Script for evaluating abundance")

    parser.add_argument("--mode", type=str, default=None, help="Mode of operation")
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="Path to checkpoint file"
    )
    parser.add_argument(
        "--abundance_map_path", type=str, default=None, help="Path to abundance map"
    )
    parser.add_argument(
        "--predicted_abundance_path", type=str, default=None, help="Path to results"
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

    return parser


@torch.no_grad()
def main():
    args = create_parser().parse_args()

    data = load_data(args)
    if not args.mode == "disentangler":
        state_map_time_series = get_state_all_data(data)[242:]
    else:
        state_map = (
            get_state_all_data(data).transpose(0, -1).transpose(0, -2).transpose(0, -3)
        )

    time_range, x_range, y_range = data.shape[0], data.shape[2], data.shape[3]
    abundance_map = np.load(args.abundance_map_path)

    match args.mode:
        case "koopman_ae":
            model = load_model(args)
            model.to(args.device)
            model.eval()

            mode_amplitude_map = extract_mode_amplitude_map(
                args, data, x_range, y_range
            )
            clusterizer = Clusterizer()
            predicted_abundance_map = clusterizer.proba_gmm(
                mode_amplitude_map, 4, "full"
            )
            classif_abundance_map = np.zeros_like(predicted_abundance_map)
            classif_abundance_map[abundance_map > 0.6] = 1
            rmse, good_permutation = compute_abundance_rmse(
                classif_abundance_map, predicted_abundance_map
            )
            print(rmse)
            print(good_permutation)
            fig, ax = plt.subplots(2, 4)
            ax[0, 0].imshow(predicted_abundance_map[:, :, good_permutation[0]])
            ax[1, 0].imshow(classif_abundance_map[:, :, 0])
            ax[0, 1].imshow(predicted_abundance_map[:, :, good_permutation[1]])
            ax[1, 1].imshow(classif_abundance_map[:, :, 1])
            ax[0, 2].imshow(predicted_abundance_map[:, :, good_permutation[2]])
            ax[1, 2].imshow(classif_abundance_map[:, :, 2])
            ax[0, 3].imshow(predicted_abundance_map[:, :, good_permutation[3]])
            ax[1, 3].imshow(classif_abundance_map[:, :, 3])
            plt.show()

        case "disentangler":
            model = load_model(args)
            model.to(args.device)
            model.eval()

            predicted_abundance_map = np.zeros((x_range, y_range, 4))
            for i in tqdm(range(x_range)):
                predicted_abundance_map[i, :, :] = (
                    model.get_abundance(
                        torch.tensor(state_map[i, :, :], dtype=torch.float32).to(
                            args.device
                        )
                    )
                    .cpu()
                    .detach()
                    .numpy()
                )
        case "cpd":
            factor = np.load(args.predicted_abundance_path)

            predicted_abundance_map = factor.reshape(x_range, y_range, -1)
            for x in range(x_range):
                for y in range(y_range):
                    for i in range(4):
                        predicted_abundance_map[x, y, i] = predicted_abundance_map[
                            x, y, i
                        ] / (sum(predicted_abundance_map[x, y]) + 1e-9)

        case "dmd":
            dmd = DynamicalModeDecomposition()
            eigenvalues, eigenvectors, initial_amplitudes = dmd.compute_dmd(data)
            initial_amplitudes = initial_amplitudes[eigenvalues.imag >= 0].reshape(
                500, 500, -1
            )

        case "dynamical_unmixing":
            predicted_abundance_map = np.load(args.predicted_abundance_path).reshape(
                x_range, y_range, -1
            )
            for x in range(x_range):
                for y in range(y_range):
                    for i in range(4):
                        predicted_abundance_map[x, y, i] = predicted_abundance_map[
                            x, y, i
                        ] / (sum(predicted_abundance_map[x, y]) + 1e-9)

        case _:
            raise ValueError(f"{args.mode} mode not recognized")

    rmse, good_permutation = compute_abundance_rmse(
        abundance_map, predicted_abundance_map
    )
    print(f"RMSE: {rmse}")

    plt.imshow(
        np.sqrt(
            np.mean(
                (abundance_map - predicted_abundance_map[:, :, good_permutation]) ** 2,
                axis=2,
            )
        )
    )
    plt.colorbar()
    plt.show()


def compute_abundance_rmse(
    abundance_map: np.ndarray, predicted_abundance_map: np.ndarray
):
    """
    Compute the MSE for all permutation of the abundance maps

    Args:
        abundance_map (np.ndarray): _description_
        predicted_abundance_map (np.ndarray): _description_

    Returns:
        mse (float): _description_
    """
    rmse = 1e9
    permutation_endmember = list(permutations(range(4)))
    good_permutation = None
    for permutation in permutation_endmember:
        rmse_permutation = np.sqrt(
            np.mean((abundance_map - predicted_abundance_map[:, :, permutation]) ** 2)
        )
        if rmse_permutation < rmse:
            rmse = rmse_permutation
            good_permutation = permutation

    return rmse, good_permutation


if __name__ == "__main__":
    main()
