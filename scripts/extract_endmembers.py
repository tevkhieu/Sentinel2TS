import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from pysptools.eea import NFINDR
from sklearn.linear_model import LinearRegression

from sentinel2_ts.dataset.process_data import scale_data
from sentinel2_ts.utils.visualize import plot_single_spectral_signature


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None, help="Path to the data")
    parser.add_argument(
        "--clipping", type=bool, default=True, help="Clipping the data or not"
    )
    parser.add_argument(
        "--num_endmembers", type=int, default=5, help="Number of endmembers to extract"
    )
    parser.add_argument("--time", type=int, default=0, help="Time step to visualize")
    parser.add_argument(
        "--save_folder", type=str, default=None, help="Path to save the endmembers"
    )
    return parser


def main():
    args = create_argparser().parse_args()
    os.makedirs(args.save_folder, exist_ok=True)
    save_path = os.path.join(args.save_folder, "endmembers.npy")
    data = np.load(args.data_path)
    data = scale_data(data, clipping=args.clipping)[args.time].transpose(1, 2, 0)

    endmember_extractor = NFINDR()
    endmembers = endmember_extractor.extract(data[37:, 33:], args.num_endmembers)
    for i in range(args.num_endmembers):
        endmembers[i][:] = endmembers[i][[0, 1, 2, 6, 3, 4, 5, 7, 8, 9]]

    np.save(save_path, endmembers)


if __name__ == "__main__":
    main()
