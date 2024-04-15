import matplotlib.pyplot as plt
import numpy as np


def visualize_spectral_signature(ax, eigenvectors, eigenvalues):
    wavelength = [
        492.4,
        559.8,
        664.6,
        704.1,
        740.5,
        782.8,
        832.8,
        864.7,
        1613.7,
        2202.4,
    ]
    band_order = [0, 1, 2, 6, 3, 4, 5, 7, 8, 9]
    for i in range(eigenvectors.shape[1]):
        ax.plot(
            wavelength,
            np.real(eigenvectors[band_order, i]),
            label=f"Period: {5 * 2 * np.pi/ np.angle(eigenvalues[i]): .4f} days",
        )
    ax.legend()
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Amplitude")
    plt.show()


def axes_off(ax):
    for a in ax:
        a.axis("off")
