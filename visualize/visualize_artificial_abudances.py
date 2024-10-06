import argparse
import numpy as np
import matplotlib.pyplot as plt


def create_argparser():
    parser = argparse.ArgumentParser(description="Visualize Abundance Maps")
    parser.add_argument(
        "--abundance_map_path",
        default="results/single_unmixer/fontainebleau_interpolated/predicted_abundance_map.npy",
        type=str,
        help="Path to the abundance map",
    )

    return parser


def main():
    args = create_argparser().parse_args()

    abundance_map = np.load(args.abundance_map_path)
    abundance_map = abundance_map / (np.sum(abundance_map, axis=-1, keepdims=True) + 1e-6)
    if len(abundance_map.shape) > 3:
        abundance_map = abundance_map[0].transpose(1, 2, 0)
    fig, ax = plt.subplots(1, 4)
    plt.subplots_adjust(bottom=0.25)
    for i in range(4):
        im = ax[i].imshow(abundance_map[:, :, i], vmax=1, vmin=0, cmap="viridis")
        ax[i].set_title(f"Endmember {i}")
        ax[i].axis("off")

    plt.colorbar(im, ax=ax.ravel().tolist())
    plt.show()


if __name__ == "__main__":
    main()
