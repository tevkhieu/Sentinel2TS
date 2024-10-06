import argparse
import numpy as np
import matplotlib.pyplot as plt
from sentinel2_ts.utils.visualize import plot_single_spectral_signature


def create_argparser():
    parser = argparse.ArgumentParser(description="Visualize endmembers")
    parser.add_argument(
        "--emdmembers_path",
        default="results/single_unmixer/fontainebleau_interpolated/endmembers.npy",
        type=str,
        help="Path to the endmembers",
    )

    return parser


def main():
    args = create_argparser().parse_args()

    endmembers = np.load(args.emdmembers_path)[:, :]
    print(endmembers.shape)
    fig, ax = plt.subplots(1, endmembers.shape[0], figsize=(30, 10))
    plt.subplots_adjust(bottom=0.25)
    for i in range(endmembers.shape[0]):
        plot_single_spectral_signature(ax[i], endmembers[i])
        ax[i].set_title(f"Endmember {i}")

    plt.show()


if __name__ == "__main__":
    main()
