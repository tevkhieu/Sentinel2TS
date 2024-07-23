import numpy as np
import scipy.io as sio
from tqdm import tqdm
import matplotlib.pyplot as plt

data = sio.loadmat("data/moffett1.mat")
im = data["im"].astype("double")
imrgb = data["imrgb"]

im = im[1:200, 1:200, :]
im = np.delete(im, slice(200, 224), axis=2)  # delete noisy or corrupted bands
im = np.delete(im, slice(142, 150), axis=2)

im = (im - np.amin(im)) / (np.amax(im) - np.amin(im))  # normalize data

im = im[:, :, [9, 16, 24, 30, 34, 38, 43, 46, 121, 180]]  # sample sentinel2 bands
im = im[:, :, [0, 1, 2, 4, 5, 6, 3, 7, 8, 9]]

abundance_map = np.load("data/true_abundance_map.npy")

data = np.zeros((343, 192, im.shape[0], im.shape[1]))

abundance_mask = abundance_map > 0.3


for x in range(im.shape[0]):
    for y in range(im.shape[1]):
        if abundance_mask[x, y, 0]:
            data[:, :, x, y] = (
                np.exp(-np.array(range(343)) / 73).reshape(-1, 1)
                * im[x, y, :].reshape(1, 192)
                * abundance_map[x, y, 0]
            )
        elif abundance_mask[x, y, 1]:
            data[:, :, x, y] = (
                ((np.sin((np.array(range(343)) / 73) * 2 * np.pi) + 1) / 2).reshape(
                    -1, 1
                )
                * im[x, y, :].reshape(1, 192)
                * abundance_map[x, y, 1]
            )
        elif abundance_mask[x, y, 2]:
            data[:, :, x, y] = (
                ((np.sin((np.array(range(343)) / 1920) * 2 * np.pi) + 1) / 2).reshape(
                    -1, 1
                )
                * im[x, y, :].reshape(1, 192)
                * abundance_map[x, y, 2]
            )
        elif abundance_mask[x, y, 3]:
            data[:, :, x, y] = (
                ((np.sin((np.array(range(343)) / 730) * 2 * np.pi) + 1) / 2).reshape(
                    -1, 1
                )
                * im[x, y, :].reshape(1, 192)
                * abundance_map[x, y, 3]
            )

np.save("data/synthetic_data_bis.npy", data)
