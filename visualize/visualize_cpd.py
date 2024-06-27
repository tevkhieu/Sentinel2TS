import os
import argparse

import numpy as np
import tensorly as tl
import matplotlib.pyplot as plt

from sentinel2_ts.utils.visualize import plot_single_spectral_signature


def create_arg_parse():
    parser = argparse.ArgumentParser(description="Visualize CPD")
    parser.add_argument(
        "--results_path", type=str, default=None, help="Path to the results"
    )
    parser.add_argument(
        "--nb_cols",
        type=int,
        default=4,
        help="Number of columns in the plot",
    )
    parser.add_argument("--x_range", type=int, default=199, help="Range of x")
    parser.add_argument("--y_range", type=int, default=199, help="Range of y")
    parser.add_argument("--time_range", type=int, default=343, help="Range of time")
    parser.add_argument("--nb_band", type=int, default=10, help="Number of bands")
    return parser


def main():
    tl.set_backend("pytorch")

    args = create_arg_parse().parse_args()
    factors = []
    factors.append(np.load(os.path.join(args.results_path, "time.npy")))
    factors.append(np.load(os.path.join(args.results_path, "specters.npy")))
    factors.append(np.load(os.path.join(args.results_path, "abundance_map.npy")))
    x_range = args.x_range
    y_range = args.y_range

    predicted_abundance_map = factors[2].reshape(x_range, y_range, -1)
    for x in range(x_range):
        for y in range(y_range):
            for i in range(4):
                predicted_abundance_map[x, y, i] = predicted_abundance_map[x, y, i] / (
                    sum(predicted_abundance_map[x, y]) + 1e-9
                )

    fig, ax = plt.subplots(3, args.nb_cols)
    fig.subplots_adjust(
        bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0.02, hspace=0.02
    )

    for i in range(args.nb_cols):
        ax[2, i].plot(range(0, 343 * 5, 5), factors[0][:, i])
        plot_single_spectral_signature(ax[0, i], factors[0][:, i])
        ax[1, i].axis("off")
        im = ax[1, i].imshow(predicted_abundance_map[:, :, i], vmin=0, vmax=1)

    cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
    fig.colorbar(im, cax=cb_ax)
    plt.show()

    # reconstruction = np.zeros_like(data)
    # for x in range(x_range):
    #     for y in range(y_range):
    #         for i in range(rank):
    #             reconstruction[:, :, x, y] = (
    #                 weight[i]
    #                 * factors[0][:, i].reshape(-1, 1)
    #                 * factors[1][:, i].reshape(1, -1)
    #                 * factors[2].reshape(x_range, y_range, rank)[x, y, i]
    #             )

    # print((np.mean((data - reconstruction) ** 2)))
    # plt.imshow(np.sqrt(np.mean((data - reconstruction) ** 2, axis=(0, 1))))
    # plt.colorbar()
    # plt.show()


if __name__ == "__main__":
    main()
