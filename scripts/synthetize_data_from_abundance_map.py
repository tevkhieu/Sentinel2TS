import numpy as np
import scipy.io as sio
from tqdm import tqdm
import matplotlib.pyplot as plt
from sentinel2_ts.runners import DatasetGenerator


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

data = np.zeros((343, 10, im.shape[0], im.shape[1]))

abundance_mask = abundance_map > 0.3
artificial_dataset = DatasetGenerator()

concrete_path = "data/endmembers/usgs/ASCIIdata_splib07b_rsSentinel2/ChapterA_ArtificialMaterials/S07SNTL2_Cadmium_orange_0_GDS786_ASDFRa_AREF.txt"
green_grass_path = "data/endmembers/usgs/ASCIIdata_splib07b_rsSentinel2/ChapterV_Vegetation/S07SNTL2_Grass_dry.4+.6green_AMX27_BECKa_AREF.txt"
oak_leaf_path = "data/endmembers/usgs/ASCIIdata_splib07b_rsSentinel2/ChapterV_Vegetation/S07SNTL2_Oak_Oak-Leaf-1_fresh_ASDFRa_AREF.txt"
cobalt_blue_path = "data/endmembers/usgs/ASCIIdata_splib07b_rsSentinel2/ChapterA_ArtificialMaterials/S07SNTL2_Cobalt_blue_GDS790_ASDFRa_AREF.txt"
concrete = artificial_dataset.extract_specter(concrete_path)
green_grass = artificial_dataset.extract_specter(green_grass_path)
oak_leaf = artificial_dataset.extract_specter(oak_leaf_path)
cobalt_blue = artificial_dataset.extract_specter(cobalt_blue_path)

endmembers = np.array([concrete, green_grass, oak_leaf, cobalt_blue])

for x in range(im.shape[0]):
    for y in range(im.shape[1]):
        if abundance_mask[x, y, 0]:
            data[:, :, x, y] = (
                np.exp(-np.array(range(343)) / 73).reshape(-1, 1)
                * concrete.reshape(1, 10)
                * abundance_map[x, y, 0]
            )
        elif abundance_mask[x, y, 1]:
            data[:, :, x, y] = (
                ((np.sin((np.array(range(343)) / 73) * 2 * np.pi) + 1) / 2).reshape(
                    -1, 1
                )
                * green_grass.reshape(1, 10)
                * abundance_map[x, y, 1]
            )
        elif abundance_mask[x, y, 2]:
            data[:, :, x, y] = (
                ((np.sin((np.array(range(343)) / 1920) * 2 * np.pi) + 1) / 2).reshape(
                    -1, 1
                )
                * oak_leaf.reshape(1, 10)
                * abundance_map[x, y, 2]
            )
        elif abundance_mask[x, y, 3]:
            data[:, :, x, y] = (
                ((np.sin((np.array(range(343)) / 730) * 2 * np.pi) + 1) / 2).reshape(
                    -1, 1
                )
                * cobalt_blue.reshape(1, 10)
                * abundance_map[x, y, 3]
            )

np.save("data/synthetic_data_no_water.npy", data)
np.save("data/endmembers/synthetic_endmembers.npy", endmembers)
