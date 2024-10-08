import argparse
import os
from itertools import permutations
import numpy as np
import torch
from tqdm import tqdm

from sentinel2_ts.runners import Clusterizer
from sentinel2_ts.utils.load_model import load_model
from sentinel2_ts.utils.process_data import get_state_time_series, get_state


def create_parser():
    parser = argparse.ArgumentParser(description="Script for evaluating abundance")

    parser.add_argument("--mode", type=str, default=None, help="Mode of operation")
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="Path to checkpoint file"
    )
    parser.add_argument(
        "--dataset_path", type=str, default=None, help="Path to dataset folder"
    )
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
        "--save_folder", type=str, default=None, help="Folder for results"
    )
    parser.add_argument("--size", type=int, default=20, help="Size of the data")
    parser.add_argument("--maximum_x", type=int, default=199, help="Maximum x value")
    parser.add_argument("--maximum_y", type=int, default=199, help="Maximum y value")
    parser.add_argument("--minimum_x", type=int, default=0, help="Minimum x value")
    parser.add_argument("--minimum_y", type=int, default=0, help="Minimum y value")
    parser.add_argument("--time_range", type=int, default=343, help="Time range")
    parser.add_argument("--data_mode", type=str, default=None, help="Data mode")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of classes")
    parser.add_argument("--abundance_mode", type=str, default="conv", help="Abundance mode")
    parser.add_argument("--disentangler_mode", type=str, default="specter", help="Disentangler mode")
    parser.add_argument("--double_decoder", type=bool, default=False, help="Double decoder")
    
    return parser


@torch.no_grad()
def main():
    args = create_parser().parse_args()
    x_range = args.maximum_x - args.minimum_x
    y_range = args.maximum_y - args.minimum_y
    match args.mode:
        case "koopman_ae":
            model = load_model(args)
            model.to(args.device)
            model.eval()

            matrix_k = torch.load(args.path_matrix_k)
            matrix_k = matrix_k.cpu().detach()
            _, eigenvectors = torch.linalg.eig(matrix_k)

            mode_amplitude_map = np.zeros((x_range, y_range, eigenvectors.shape[0]))
            inverse_eigenvectors = torch.pinverse(eigenvectors).to(args.device)
            for i in tqdm(range(x_range)):
                for j in range(y_range):
                    input_data = (
                        get_state(
                            np.load(
                                os.path.join(args.dataset_path, f"{i:03}_{j:03}.npy")
                            ),
                            1,
                        )
                    ).to(args.device)
                    mode_amplitude_map[i, j, :] = (
                        torch.real(
                            inverse_eigenvectors.to(torch.complex64)
                            @ model.encode(input_data).to(torch.complex64)
                        )
                        .cpu()
                        .detach()
                        .numpy()
                    )

            clusterizer = Clusterizer()
            predicted_abundance_map = clusterizer.proba_gmm(
                mode_amplitude_map, 4, "full"
            ).reshape(x_range, y_range, 4)

        case "disentangler":
            model = load_model(args)
            model.to(args.device)
            model.eval()

            predicted_abundance_map = np.zeros((x_range, y_range, args.num_classes))
            for i in tqdm(range(x_range)):
                for j in range(y_range):
                    if args.data_mode == "time_series":
                        input_data = (
                            torch.Tensor(
                                np.load(
                                    os.path.join(
                                        args.dataset_path, f"{i:03}_{j:03}.npy"
                                    )
                                ).transpose(1, 0)
                            )
                            .unsqueeze(0)[:, :, 1:]
                            .to(args.device)
                        )
                    else:
                        input_data = (
                            get_state_time_series(
                                np.load(
                                    os.path.join(
                                        args.dataset_path, f"{i:03}_{j:03}.npy"
                                    )
                                ),
                                1,
                                342,
                            )
                            .transpose(0, 1)
                            .unsqueeze(0)
                            .to(args.device)
                        )
                    predicted_abundance_map[i, j, :] = (
                        model.get_abundance(input_data).cpu().detach().numpy()
                    )

        case "koopman_unmixer":
            model = load_model(args)
            model.to(args.device)
            model.eval()

            predicted_abundance_map = np.zeros((x_range, y_range, 4))
            for i in tqdm(range(x_range)):
                for j in range(y_range):
                    input_data = (
                        get_state(
                            np.load(
                                os.path.join(args.dataset_path, f"{i:03}_{j:03}.npy")
                            ),
                            1,
                        )
                        .unsqueeze(0)
                        .to(args.device)
                    )
                    predicted_abundance_map[i, j, :] = (
                        model.get_abundance(input_data).cpu().detach().numpy()
                    )

        case _:
            raise ValueError(f"{args.mode} mode not recognized")

    os.makedirs(args.save_folder, exist_ok=True)
    np.save(
        os.path.join(args.save_folder, "predicted_abundance_map.npy"),
        predicted_abundance_map,
    )


if __name__ == "__main__":
    main()
