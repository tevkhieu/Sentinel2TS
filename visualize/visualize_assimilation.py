import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def main():
    initial_abundance_map = np.exp(np.load("initial_abundance_map.npy"))
    initial_variation_map = np.load("initial_variation_map.npy")

    for i in range(500):
        for j in range(500):
            initial_abundance_map[i, j, :] = initial_abundance_map[i, j, :] / np.sum(
                initial_abundance_map[i, j, :]
            )

    fig, ax = plt.subplots(ncols=2)
    plt.subplots_adjust(left=0.1, bottom=0.25)
    time_slider = Slider(
        plt.axes([0.1, 0.1, 0.8, 0.03]), "Time", 0, 342, valinit=0, valstep=1
    )
    endmember_slider = Slider(
        plt.axes([0.1, 0.15, 0.8, 0.03]), "Endmember", 0, 4, valinit=0, valstep=1
    )
    abundance_map = ax[0].imshow(initial_abundance_map[:, :, 0], vmin=0, vmax=1)
    variation_map = ax[1].imshow(initial_variation_map[0, :, :, 0], vmin=0, vmax=1)

    def update(val):
        time = int(time_slider.val)
        endmember = int(endmember_slider.val)
        ax[0].imshow(initial_abundance_map[:, :, endmember], vmin=0, vmax=1)
        ax[1].imshow(initial_variation_map[time, :, :, endmember], vmin=0, vmax=1)
        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    endmember_slider.on_changed(update)
    ax[0].set_title("Abundance Map")
    ax[1].set_title("Variation Map")
    plt.colorbar(abundance_map)
    plt.colorbar(variation_map)
    plt.show()


if __name__ == "__main__":
    main()
