import numpy as np
import matplotlib.pyplot as plt

from sentinel2_ts.utils.visualize import plot_single_spectral_signature


def main():
    abundance_map = np.load("data/true_abundance_map.npy")
    endmembers = np.load("data/endmembers/california/endmembers_california.npy")
    endmembers = endmembers[:, [9, 16, 24, 30, 34, 38, 43, 46, 121, 180]]
    endmembers = endmembers[:, [0, 1, 2, 4, 5, 6, 3, 7, 8, 9]]
    endmembers = endmembers.T

    time_series = np.zeros((343, 4))
    time_series[:, 0] = np.exp(-np.array(range(343)) / 73)
    time_series[:, 1] = (np.sin((np.array(range(343)) / 73) * 2 * np.pi) + 1) / 2
    time_series[:, 2] = (np.sin((np.array(range(343)) / 100) * 2 * np.pi) + 1) / 2
    time_series[:, 3] = (np.sin((np.array(range(343)) / 730) * 2 * np.pi) + 1) / 2

    fig, ax = plt.subplots(3, 4)
    fig.subplots_adjust(
        bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0.02, hspace=0.02
    )

    for i in range(4):
        ax[2, i].plot(range(0, 343 * 5, 5), time_series[:, i])
        plot_single_spectral_signature(ax[0, i], endmembers[:, i])
        ax[1, i].axis("off")
        im = ax[1, i].imshow(abundance_map[:, :, i].reshape(199, 199), vmin=0, vmax=1)

    cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
    fig.colorbar(im, cax=cb_ax)
    plt.show()


if __name__ == "__main__":
    main()
