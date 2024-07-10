import os
import argparse

import numpy as np
import tensorly as tl
from tensorly.decomposition import CP_NN_HALS

from sentinel2_ts.utils.load_model import load_data


def create_arg_parse():
    parser = argparse.ArgumentParser(description="Compute CPD")
    parser.add_argument(
        "--data_path", type=str, default=None, help="Path to the data to visualize"
    )
    parser.add_argument(
        "--clipping",
        type=bool,
        default=True,
        help="Whether to clip the data or not",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device on which to perform the computation",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help="Number of columns in the plot",
    )
    parser.add_argument("--scale_data", type=bool, help="Scale the data")
    parser.add_argument(
        "--save_folder", type=str, default=None, help="Folder to save the results"
    )

    return parser


def main():
    tl.set_backend("pytorch")

    args = create_arg_parse().parse_args()
    data = load_data(args)
    time_range, nb_band, x_range, y_range = data.shape

    data = data.reshape(time_range, nb_band, -1)
    cpd = CP_NN_HALS(rank=args.rank, n_iter_max=100, verbose=True)
    weights, factors = cpd.fit_transform(tl.tensor(data))
    os.makedirs(args.save_folder, exist_ok=True)
    np.save(os.path.join(args.save_folder, "weights.npy"), weights.numpy())
    for factor, name in zip(factors, ["time", "specters", "abundance_map"]):
        if name == "abundance_map":
            factor = factor.reshape(x_range, y_range, -1)
        np.save(os.path.join(args.save_folder, f"{name}.npy"), factor.numpy())


if __name__ == "__main__":
    main()
