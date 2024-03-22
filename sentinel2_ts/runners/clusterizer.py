from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# print(data_small.shape)

# reflectances = data_small.transpose(3, 1, 2, 0)
# initial_time = 1
# algo = KMeans(n_clusters=3, random_state=0)
# algo = GaussianMixture(n_components=3)
# times = [1, 11, 21, 31, 41, 51, 61, 71]

# initial_state_total = torch.Tensor(reflectances[:, :, :, times[0]]).to(device)
# initial_state_total = torch.cat(
#     (
#         initial_state_total,
#         torch.Tensor(
#             reflectances[:, :, :, times[0]] - reflectances[:, :, :, times[0] - 1]
#         ).cuda(),
#     )
# ).permute((1, 2, 0))
# X = model.encode(initial_state_total).reshape((22500, 32))
# for t in times[1:]:
#     initial_state_total = torch.Tensor(reflectances[:, :, :, t]).to(device)
#     initial_state_total = torch.cat(
#         (
#             initial_state_total,
#             torch.Tensor(
#                 reflectances[:, :, :, t] - reflectances[:, :, :, t - 1]
#             ).cuda(),
#         )
#     ).permute((1, 2, 0))
#     print(X.shape)
#     X = torch.cat((X, model.encode(initial_state_total).reshape((22500, 32))), dim=1)
# X = X.cpu().detach().numpy()
# clustering = algo.fit_predict(X)

# initial_state_total = torch.Tensor(reflectances[:, :, :, times[0]]).to(device)
# initial_state_total = torch.cat(
#     (
#         initial_state_total,
#         torch.Tensor(
#             reflectances[:, :, :, times[0]] - reflectances[:, :, :, times[0] - 1]
#         ).cuda(),
#     )
# )
# for t in times[1:]:
#     print(initial_state_total.shape)
#     initial_state_total = torch.cat(
#         (initial_state_total, torch.Tensor(reflectances[:, :, :, t]).to(device))
#     )
#     initial_state_total = torch.cat(
#         (
#             initial_state_total,
#             torch.Tensor(
#                 reflectances[:, :, :, t] - reflectances[:, :, :, t - 1]
#             ).cuda(),
#         )
#     )
# initial_state_total = initial_state_total.permute((1, 2, 0))

# X = initial_state_total
# print(X.shape)
# X = X.flatten(0, 1).cpu().detach().numpy()
# base_clustering = algo.fit_predict(X)

# plt.figure(figsize=(20, 12))
# plt.subplot(131)
# plt.imshow(np.flip(reflectances[:3, :, :, initial_time], 0).transpose((1, 2, 0)) * 3)
# plt.title(label="Sample image (RGB, beginning of the time series)")
# plt.axis("off")
# plt.subplot(132)
# plt.imshow(base_clustering.reshape((150, 150)))
# plt.title(label="Clustering of the pixels on their reflectance time series")
# plt.axis("off")
# plt.subplot(133)
# plt.imshow(clustering.reshape((150, 150)))
# plt.title("Clustering of the pixels on the encoding of their reflectance time series")
# plt.axis("off")


class Clusterizer:
    def __init__(self):
        pass

    def find_best_silhouette_score_index(self, data: ArrayLike) -> float:
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
            nb_clusters = self.find_best_silhouette_score_index(data_cluster)
        k_means = KMeans(n_clusters=nb_clusters, random_state=0, n_init="auto")
        return k_means.fit_predict(data_cluster).reshape((x, y))
