from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.manifold import TSNE
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class Clusterizer:
    def __init__(self):
        pass

    def __index_best_silhoutte_score(self, data: ArrayLike) -> float:
        """
        Find the best number of clusters for the data using the silhouette score

        Args:
            data (ArrayLike): Data to cluster

        Returns:
            float: Best silhouette score
        """
        best_score = -1
        nb_clusters = -1
        for i in tqdm(range(2, 10)):
            algo = KMeans(n_clusters=i, random_state=0)
            clustering = algo.fit_predict(data)
            score = silhouette_score(data, clustering)
            if score > best_score:
                best_score = score
                nb_clusters = i
        print(
            f"The best number of cluster according to the silhouette score is {nb_clusters} with a score of {best_score}"
        )

        return nb_clusters

    def clusterize_kmeans(self, data: ArrayLike, nb_clusters: int = None) -> ArrayLike:
        """
        Clusterize the data using KMeans

        Args:
            data (ArrayLike): Data to cluster

        Returns:
            ArrayLike: Clustering of the data
        """
        x, y, bands = data.shape
        data_cluster = data.reshape((-1, bands))
        if nb_clusters is None:
            nb_clusters = self.__index_best_silhoutte_score(data_cluster)
        k_means = KMeans(n_clusters=nb_clusters, random_state=0, n_init="auto")
        return k_means.fit_predict(data_cluster).reshape((x, y))

    def clusterize_gmm(
        self,
        data: ArrayLike,
        nb_components: int = None,
        covariance_type: str = None,
    ) -> ArrayLike:
        """
        Clusterize the data using Gaussian Mixture Model

        Args:
            data (ArrayLike): Data to cluster

        Returns:
            ArrayLike: Clustering of the data
        """
        x, y, bands = data.shape
        data_cluster = data.reshape((-1, bands))
        if nb_components is None or covariance_type is None:
            nb_components, covariance_type = self.__grid_search_gmm(data_cluster)

        gmm = GaussianMixture(
            n_components=nb_components,
            covariance_type=covariance_type,
            random_state=0,
            max_iter=1000,
        )

        return gmm.fit_predict(data_cluster).reshape((x, y))

    def __grid_search_gmm(self, data: ArrayLike) -> tuple[int, str]:
        param_grid = {
            "n_components": range(1, 7),
            "covariance_type": ["spherical", "tied", "diag", "full"],
        }
        grid_search = GridSearchCV(
            GaussianMixture(), param_grid=param_grid, scoring=self.gmm_bic_score
        )

        grid_search.fit(data)

        return (
            grid_search.best_params_["n_components"],
            grid_search.best_params_["covariance_type"],
        )

    @staticmethod
    def gmm_bic_score(estimator, X):
        """Callable to pass to GridSearchCV that will use the BIC score."""
        # Make it negative since GridSearchCV expects a score to maximize
        return -estimator.bic(X)

    def clusterize_dbscan(self, data: ArrayLike) -> ArrayLike:
        """
        Clusterize the data using DBSCAN

        Args:
            data (ArrayLike): Data to cluster

        Returns:
            ArrayLike: Clustering of the data
        """
        x, y, bands = data.shape
        data_cluster = data.reshape((-1, bands))
        clusterer = DBSCAN(metric="cosine")
        return clusterer.fit_predict(data_cluster).reshape((x, y))

    def proba_gmm(
        self,
        data: ArrayLike,
        nb_components: int = None,
        covariance_type: str = None,
    ) -> ArrayLike:
        """
        Clusterize the data using Gaussian Mixture Model

        Args:
            data (ArrayLike): Data to cluster

        Returns:
            ArrayLike: Clustering of the data
        """
        x, y, bands = data.shape
        data_cluster = data.reshape((-1, bands))
        if nb_components is None or covariance_type is None:
            nb_components, covariance_type = self.__grid_search_gmm(data_cluster)

        gmm = GaussianMixture(
            n_components=nb_components,
            covariance_type=covariance_type,
            random_state=0,
            max_iter=1000,
        )
        gmm.fit(data_cluster)
        return gmm.predict_proba(data_cluster).reshape((x, y, nb_components))

    def clusterize_tsne(self, data: ArrayLike) -> ArrayLike:
        """
        Clusterize the data using t-SNE

        Args:
            data (ArrayLike): Data to cluster

        Returns:
            ArrayLike: Clustering of the data
        """

        x, y, bands = data.shape
        data_cluster = data.reshape((-1, bands))
        tsne = TSNE(n_components=2, random_state=0, perplexity=10, learning_rate=500)
        return tsne.fit_transform(data_cluster)
