import os
import argparse
import numpy as np
from sentinel2_ts.utils.process_data import scale_data


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Extract time series from images')

    parser.add_argument('--data_path', type=str, default=None, help='Path to the data')
    parser.add_argument('--dataset_name', type=str, default=None, help='name of the dataset')
    parser.add_argument('--clipping', type=bool, default=True, help='Whether to clip the data to 1 or not')

    return parser

def main():
    args = create_arg_parser().parse_args()
    dataset_dir = os.path.join("datasets", args.dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    data = np.load(args.data_path)
    data = scale_data(data, args.clipping)
    for x in range(250, 400):
        for y in range(250, 400):
            np.save(os.path.join(dataset_dir, f"{x:03}_{y:03}.npy"), data[..., x, y])

if __name__ == "__main__":
    main()