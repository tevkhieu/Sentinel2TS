import argparse
import numpy as np
import tensorly as tl
import matplotlib.pyplot as plt
import torch
from tensorly.decomposition import CP_NN_HALS
from sentinel2_ts.data.process_data import scale_data


def create_arg_parse():
    parser = argparse.ArgumentParser(description="Visualize CPD")
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
        "--nb_rows",
        type=int,
        default=1,
        help="Number of rows in the plot",
    )
    parser.add_argument(
        "--nb_cols",
        type=int,
        default=4,
        help="Number of columns in the plot",
    )
    return parser


def main():
    tl.set_backend("pytorch")

    args = create_arg_parse().parse_args()

    data = scale_data(np.load(args.data_path), clipping=args.clipping).reshape(
        343, 10, -1
    )

    rank = args.nb_rows * args.nb_cols

    decomp = CP_NN_HALS(rank)
    weight, factors = decomp.fit_transform(tl.tensor(data).to(args.device))
    factors[2] /= tl.max(factors[2])

    fig, ax = plt.subplots(args.nb_rows, args.nb_cols)
    fig.subplots_adjust(
        bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0.02, hspace=0.02
    )

    if args.nb_rows > 1:
        for i in range(rank):
            ax[i // args.nb_cols, i % args.nb_cols].axis("off")
            im = ax[i // args.nb_cols, i % args.nb_cols].imshow(
                factors[2][:, i].reshape(500, 500), vmin=0, vmax=1
            )
    else:
        for i in range(rank):
            ax[i].axis("off")
            im = ax[i].imshow(factors[2][:, i].reshape(500, 500), vmin=0, vmax=1)

    cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
    fig.colorbar(im, cax=cb_ax)
    plt.show()


if __name__ == "__main__":
    main()
