import matplotlib.pyplot as plt
import numpy as np


def plot_all_spectral_signatures(ax, eigenvectors):
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
            # label=f"Period: {5 * 2 * np.pi/ np.angle(eigenvalues[i]): .4f} days",
        )

def plot_single_spectral_signature(ax, spectral_signature):
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
    ax.plot(wavelength, spectral_signature[band_order])

def axes_off(ax):
    for a in ax:
        a.axis("off")
