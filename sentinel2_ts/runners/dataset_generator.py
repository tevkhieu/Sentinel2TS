import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.special import digamma, polygamma


class DatasetGenerator:
    """
    Class is used to generate artificial dataset for testing purposes.
    """

    def __init__(self, index=[1, 2, 3, 8, 4, 5, 6, 9, 11, 12], time_length=343):
        self.index = index
        self.time_length = time_length

    def extract_specter(self, file_path: str) -> np.array:
        """
        Extract the specter from the file

        Args:
            file_path (str): path to the file

        Returns:
            np.array: extracted specter
        """
        f = open(file_path)
        specter = f.readlines()[1:]
        f.close()
        return np.array([float(i.split()[0]) for i in specter])[self.index]

    def generate_sin_time_series(
        self, specters: np.ndarray, amplitudes: np.ndarray, period: np.ndarray
    ) -> np.array:
        """
        Generate a sinusoidal specter

        Args:
            specters (np.ndarray): specters to generate the time series from
            amplitudes (np.ndarray): amplitudes of the specters
            period (np.ndarray): periods of the specters

        Returns:
            np.array: generated specters
        """
        generated_specter = np.zeros((self.time_length, specters.shape[1]))
        for t in range(self.time_length):
            for i in range(specters.shape[0]):
                generated_specter[t] += (
                    np.sin(2 * np.pi * t / period[i]) * specters[i] * amplitudes[i]
                    + 0.5
                )

        return generated_specter

    def generate_neg_exponential_time_series(
        self, specters: np.ndarray, amplitudes: np.ndarray, period: np.ndarray
    ):
        """
        Generate a negative exponential specter

        Args:
            specters (np.ndarray): specters to generate the time series from
            amplitudes (np.ndarray): amplitudes of the specters
            period (np.ndarray): periods of the specters

        Returns:
            np.array: generated specters
        """
        generated_specter = np.zeros((self.time_length, specters.shape[1]))
        for t in range(self.time_length):
            for i in range(specters.shape[0]):
                generated_specter[t] += (
                    np.exp(-t / period[i]) * specters[i] * amplitudes[i]
                )

        return generated_specter

    def generate_entangled_time_series(
        self, time_series: np.ndarray, alpha, size: int = None, data: np.ndarray = None
    ):
        """
        Generate data from the time series

        Args:
            time_series (np.ndarray): time series to generate the data from
        """
        distribution = np.random.dirichlet(alpha=alpha).reshape(-1, 1, 1)
        return (
            np.sum(time_series * distribution, axis=0)
            + np.random.normal(0, 1e-2, (time_series.shape[1], time_series.shape[2])),
            distribution,
        )

    def estimate_alpha(self, mode_amplitude_map: np.ndarray):
        """
        Estimate the alpha parameter of the Dirichlet distribution using maximum likelihood
        of the dirichlet distribution sampling the distribution with realization of a gmm

        Args:
            data (np.ndarray): data to estimate the alpha parameter from

        Returns:
            np.ndarray: estimated alpha parameter
        """
        gmm = GaussianMixture(
            covariance_type="full", n_components=4, max_iter=1000, random_state=0
        )
        x, y, bands = mode_amplitude_map.shape
        data_cluster = mode_amplitude_map.reshape((-1, bands))
        gmm.fit(data_cluster)
        logp = gmm.predict_proba(data_cluster).mean(0)
        alpha = np.ones(4)
        for i in range(1000):
            alpha = self.__inverse_digamma(digamma(alpha.sum()) + logp)

        return alpha

    def __inverse_digamma(self, y: np.ndarray):
        """
        Compute the inverse digamma function using Newton algorithm

        Args:
            y (np.ndarray): output of the digamma function

        Returns:
            x (np.ndarray): inverse digamma function
        """

        x = np.piecewise(
            y,
            [y >= -2.22, y < -2.22],
            [lambda x: -1 / (x + digamma(1)), lambda x: np.exp(x) + 0.5],
        )
        for _ in range(5):
            x = x - (digamma(x) - y) / (polygamma(1, x))

        return x
