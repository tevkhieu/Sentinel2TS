import argparse
import numpy as np
import matplotlib.pyplot as plt


def create_argparser():
    parser = argparse.ArgumentParser(description="Visualize Abundance Maps")
    parser.add_argument(
        "--abundance_map_path",
        default="data/true_abundance_map.npy",
        type=str,
        help="Path to the abundance map",
    )

    return parser


def main():
    args = create_argparser().parse_args()

    abundance_map = np.load(args.abundance_map_path)
    fig, ax = plt.subplots(1, 4)
    plt.subplots_adjust(bottom=0.25)
    for i in range(4):
        im = ax[i].imshow(abundance_map[:, :, i], vmin=0, vmax=1)
        ax[i].set_title(f"Endmember {i}")
        ax[i].axis("off")

    plt.colorbar(im, ax=ax.ravel().tolist())
    plt.show()


if __name__ == "__main__":
    main()
