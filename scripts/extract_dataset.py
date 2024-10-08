import os
import argparse
import numpy as np
from tqdm import tqdm

from sentinel2_ts.utils.load_model import load_data


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Extract time series from images")

    parser.add_argument("--data_path", type=str, default=None, help="Path to the data")
    parser.add_argument(
        "--dataset_name", type=str, default=None, help="name of the dataset"
    )
    parser.add_argument(
        "--clipping",
        type=bool,
        default=True,
        help="Whether to clip the data to 1 or not",
    )
    parser.add_argument(
        "--minimum_x", type=int, default=250, help="Minimal x value on the image"
    )
    parser.add_argument(
        "--maximum_x", type=int, default=400, help="Maximal x value on the image"
    )
    parser.add_argument(
        "--minimum_y", type=int, default=250, help="Minimal y value on the image"
    )
    parser.add_argument(
        "--maximum_y", type=int, default=400, help="Maximal y value on the image"
    )
    parser.add_argument("--mask", type=str, default=None, help="path to the mask file")
    parser.add_argument("--scale_data", type=bool, help="Scale the data")
    parser.add_argument(
        "--data_type", default="time_series", type=str, help="Type of data"
    )
    return parser


def main():
    args = create_arg_parser().parse_args()
    dataset_dir = os.path.join("datasets", args.dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    data = load_data(args)

    indices = range(data.shape[0])

    if args.mask is not None:
        mask = np.load(args.mask)
        indices = np.where(mask == 1)

    match args.data_type:
        case "time_series":
            for x in tqdm(range(args.minimum_x, args.maximum_x)):
                for y in range(args.minimum_y, args.maximum_y):
                    np.save(
                        os.path.join(dataset_dir, f"{x:03}_{y:03}.npy"),
                        data[indices, :, x, y],
                    )
        case "images":
            for t in indices:
                np.save(os.path.join(dataset_dir, f"{t:03}.npy"), data[t, :, args.minimum_x:args.maximum_x, args.minimum_y:args.maximum_y])


if __name__ == "__main__":
    main()
