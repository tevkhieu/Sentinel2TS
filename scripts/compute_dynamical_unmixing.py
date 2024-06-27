import os
import argparse
import numpy as np

from sentinel2_ts.utils.load_model import load_data
from sentinel2_ts.runners import DynamicalSpectralUnmixer


def create_argparser():
    parser = argparse.ArgumentParser(description="Compute Dynamical Unmixing")

    parser.add_argument(
        "--data_path", type=str, default=None, help="Path to the data to unmix"
    )
    parser.add_argument(
        "--initial_specters_path",
        type=str,
        default=None,
        help="Path to the initial specters",
    )
    parser.add_argument("--scale_data", type=bool, help="Scale the data")
    parser.add_argument(
        "--clipping", type=bool, default=True, help="Whether to clip the data or not"
    )
    parser.add_argument(
        "--save_folder", type=str, default=None, help="Folder to save the results"
    )
    parser.add_argument(
        "--max_iter", type=int, default=100, help="Maximum number of iterations"
    )
    return parser


def main():
    args = create_argparser().parse_args()

    data = load_data(args)
    initial_specters = np.load(args.initial_specters_path).T

    unmixer = DynamicalSpectralUnmixer(data, initial_specters)
    specter, abundance_map, psi = unmixer.unmix(max_iter=args.max_iter)

    os.makedirs(args.save_folder, exist_ok=True)
    np.save(os.path.join(args.save_folder, "specter.npy"), specter)
    np.save(os.path.join(args.save_folder, "abundance_map.npy"), abundance_map)
    np.save(os.path.join(args.save_folder, "psi.npy"), psi)


if __name__ == "__main__":
    main()
