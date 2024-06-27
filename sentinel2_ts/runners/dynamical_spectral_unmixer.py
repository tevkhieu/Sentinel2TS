import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from sentinel2_ts.utils.visualize import plot_single_spectral_signature


class DynamicalSpectralUnmixer:
    """
    Python implementation of Henrot et al. (2015).
    Dynamical Spectral Unmixing of Multitemporal Hyperspectral Images,
    IEEE Transactions on Image Processing, 25(7), 3219 - 3232
    """

    def __init__(self, data: np.ndarray, initial_specters: np.ndarray) -> None:
        """
        Initialize the dynamical spectral unmixer

        Args:
            data (ArrayLike): Data to unmix
        """
        self.data = data.reshape(data.shape[0], 10, -1)
        self.time_range, self.nb_bands, self.nb_pixels = self.data.shape
        self.nb_endmembers = initial_specters.shape[1]

        # Initialize the specters
        self.specters = np.array([initial_specters] * self.time_range)

        # Initialize the abundance map
        self.abundance_map = np.array(
            self.time_range * [np.eye(self.nb_endmembers, self.nb_pixels)]
        )

        # Initialize the psi matrix
        self.psi = np.array([np.eye(self.nb_endmembers)] * self.time_range)

        self.lambda_s = 1
        self.lambda_a = 1
        self.rho = 1

        # dual for specter optimization
        self.U_specters = np.zeros((self.time_range, self.nb_bands, self.nb_endmembers))

        # dual for abundance optimization
        self.U_abundance = np.zeros(
            (self.time_range, self.nb_endmembers, self.nb_pixels)
        )

    def __specters_update(self):
        """
        Update the specters according to equation 12 of
        Henrot et al. (2015) such that the specters remain positive
        Dynamical Spectral Unmixing of Multitemporal Hyperspectral Images,
        IEEE Transactions on Image Processing, 25(7), 3219 - 3232
        """
        M = np.clip(self.specters + self.U_specters, a_min=0, a_max=None)
        self.specters = (
            (self.data @ self.abundance_map.transpose(0, 2, 1))
            + self.lambda_s
            * (np.array([self.specters[0]] * self.time_range) @ self.psi)
            + self.rho * (M - self.U_specters)
        ) @ np.linalg.inv(
            self.abundance_map @ self.abundance_map.transpose(0, 2, 1)
            + (self.lambda_s + self.rho) * np.eye(self.nb_endmembers)
        )
        self.U_specters += self.specters - M

    def __psi_update(self):
        """
        Update the psi matrix according to equation 19 of
        Henrot et al. (2015).
        Dynamical Spectral Unmixing of Multitemporal Hyperspectral Images,
        IEEE Transactions on Image Processing, 25(7), 3219 - 3232
        """
        for k in range(self.time_range):
            for p in range(self.nb_endmembers):
                self.psi[k, p, p] = (self.specters[0, p].T @ self.specters[k, p]) / (
                    self.specters[0, p].T @ self.specters[0, p]
                )

    @staticmethod
    def __proj_simplex(data):
        """
        Projection of the columns of a PxN matrix on the unit simplex (with P vertices).

        Args:
            data (numpy.ndarray): The input data to be projected, assumed to be a 2D array where each column
                          represents a different vector to be projected.

        Returns:
            numpy.ndarray: The projected data, with the same shape as the input, where each column vector
                   lies on the probability simplex.
        """
        data_sorted = np.sort(data, axis=0)[
            ::-1, :
        ]  # sort rows of data array in descending order (by going through each column backwards)
        cumulative_sum = (
            np.cumsum(data_sorted, axis=0) - 1
        )  # cumulative sum of each row
        vector = (
            np.arange(np.shape(data_sorted)[0]) + 1
        )  # define vector to be divided elementwise
        divided = cumulative_sum / vector[:, None]  # perform the termwise division
        projected = np.maximum(
            data - np.amax(divided, axis=0), np.zeros(divided.shape)
        )  # projection step

        return projected

    def __a_update(self):
        """Update the abundance map using Lucas Drumetz's method"""
        Z = np.zeros((self.time_range, self.nb_endmembers, self.nb_pixels))
        for k in range(self.time_range):
            Z[k] = self.__proj_simplex(self.abundance_map[k] + self.U_abundance[k])
        self.abundance_map = np.linalg.inv(
            np.sum(np.matmul(self.specters.transpose(0, 2, 1), self.specters), axis=0)
            + self.rho * np.eye(self.nb_endmembers)
        ) @ np.sum(
            np.matmul(self.specters.transpose(0, 2, 1), self.data), axis=0
        ) + self.rho * (
            Z - self.U_abundance
        )
        self.U_abundance = self.U_abundance + self.abundance_map - Z

    def unmix(self, max_iter=200, max_iter_admm=100, tol=1e-3):
        """
        Unmix the data

        Args:
            max_iter (int, optional): Number of iteration for alternating least squares. Defaults to 200.
            max_iter_admm (int, optional): Number of iteration for ADMM. Defaults to 100.
            tol (_type_, optional): tolerance. Defaults to 1e-3.

        Returns:
            tuple: The specters, the abundance map and the psi matrix
        """

        for i in tqdm(range(max_iter)):
            for j in range(max_iter_admm):
                self.__specters_update()
            for j in range(max_iter_admm):
                self.__a_update()
            self.__psi_update()

        return self.specters, self.abundance_map, self.psi


if __name__ == "__main__":
    data = np.load("data/synthetic_data.npy")
    initial_specters = np.load(
        "data/endmembers/california/endmembers_california_sentinel2.npy"
    ).T
    unmixer = DynamicalSpectralUnmixer(data, initial_specters)
    s, a, p = unmixer.unmix(max_iter=10, max_iter_admm=100, tol=1e-3)
    print(s.shape, a.reshape(343, 4, 199, 199).shape, p.shape)
