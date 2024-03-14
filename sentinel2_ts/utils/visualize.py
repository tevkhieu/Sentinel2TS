import matplotlib.pyplot as plt


def visualize_spectral_signature(args, eigenvectors):
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
    fig, ax = plt.subplots()
    for i in range(args.rank_approximation):
        ax.plot(wavelength, eigenvectors[band_order, i], label=f"Eigenvalue {i}")
    ax.legend()
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Amplitude")
    plt.show()
