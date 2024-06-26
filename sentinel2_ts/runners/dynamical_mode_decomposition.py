import numpy as np
from numpy.typing import ArrayLike


class DynamicalModeDecomposition:
    def __init__(self) -> None:
        pass

    @staticmethod
    def compute_dmd(data: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        """
        Compute the DMD of the data

        Args:
            data (ArrayLike): Time series map

        Returns:
            Tuple[ArrayLike, ArrayLike]: eigenvectors and Lambda DMD modes and eigenvalues
        """
        time_range, band, x_range, y_range = data.shape
        data = data.reshape(time_range, -1)
        X = data[:-1, :]
        Y = data[1:, :]

        U, s, Vh = np.linalg.svd(X.T, full_matrices=False)

        r = len(s)
        S_r = np.diag(s[:r])

        A_tilde = U.T @ Y.T @ Vh.T @ np.linalg.inv(S_r)
        eigenvalues, eigenvectors = np.linalg.eig(A_tilde)

        temp = Y.T @ Vh.T @ np.diag(1 / s)

        modes = (temp @ eigenvectors) / eigenvalues[np.newaxis, :]

        initial_amplitudes = np.linalg.pinv(modes @ np.diag(eigenvalues)) @ data[
            1
        ].reshape(-1)

        return (
            eigenvalues,
            modes.reshape(time_range - 1, band, x_range, y_range),
            initial_amplitudes,
        )

    @staticmethod
    def compute_pixel_dmd(data: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        """
        Compute the DMD of the data at point x, y

        Args:
            data (ArrayLike): Time series data

        Returns:
            Tuple[ArrayLike, ArrayLike]: Phi and Lambda DMD modes and eigenvalues
        """
        x_data = data[:-1, :].T
        x_prime_data = data[1:, :].T

        # Performing Singular Value Decomposition (SVD)
        U, S, Vh = np.linalg.svd(x_data, full_matrices=False)

        # Constructing the approximation of the A matrix
        A_tilde = U.T @ x_prime_data @ Vh.T @ np.linalg.inv(np.diag(S))

        # Calculating eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(A_tilde)

        # Constructing the DMD modes Phi
        Phi = x_prime_data @ Vh.T @ np.linalg.inv(np.diag(S)) @ eigenvectors

        return Phi, eigenvalues
