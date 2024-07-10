import argparse
import os
from itertools import permutations
import numpy as np
import matplotlib.pyplot as plt
import yaml


def create_parser():
    parser = argparse.ArgumentParser(description="Script for evaluating abundance")

    parser.add_argument(
        "--abundance_map_path", type=str, default=None, help="Path to abundance map"
    )
    parser.add_argument(
        "--predicted_abundance_path", type=str, default=None, help="Path to results"
    )
    parser.add_argument("--save_folder", type=str, default=None, help="Folder to save")
    parser.add_argument("--good_permutation", type=int, nargs="+", help="Good permutation")

    return parser


def main():
    args = create_parser().parse_args()

    abundance_map = np.load(args.abundance_map_path)
    predicted_abundance_map = np.load(args.predicted_abundance_path)
    if len(predicted_abundance_map.shape) > 3:
        predicted_abundance_map = predicted_abundance_map[0].transpose(1, 2, 0)

    accuracy, good_permutation, accuracy_list = compute_accuracy(
        abundance_map, predicted_abundance_map, args.good_permutation
    )
    print(f"Accuracy: {accuracy}, Good Permutation: {good_permutation}")

    rmse, good_permutation, rmse_list = compute_abundance_rmse(
        abundance_map, predicted_abundance_map, args.good_permutation
    )

    print(f"RMSE: {rmse}, Good Permutation: {good_permutation}")

    # Save the results
    os.makedirs(args.save_folder, exist_ok=True)
    with open(os.path.join(args.save_folder, "abundance.yaml"), "w") as f:
        yaml.dump(
            {
                "RMSE": float(rmse),
                "Accuracy": float(accuracy),
                "Good Permutation": list(good_permutation),
                "RMSE List": [float(r) for r in rmse_list],
                "Accuracy List": [float(a) for a in accuracy_list],
            },
            f,
        )
    plt.imshow(
        np.sqrt(
            np.mean(
                (abundance_map - predicted_abundance_map[:, :, good_permutation]/(np.sum(predicted_abundance_map, axis=2, keepdims=True) + 1e-6)) ** 2,
                axis=2,
            )
        )
    )
    plt.title("Abundance RMSE")
    plt.axis("off")
    plt.colorbar()
    plt.savefig(os.path.join(args.save_folder, "abundance_rmse.png"))

    fig, ax = plt.subplots(2, 4, figsize=(20, 10))
    for i in range(4):
        ax[0, i].imshow(abundance_map[:, :, i], vmin=0, vmax=1)
        ax[1, i].imshow(
            predicted_abundance_map[:, :, good_permutation[i]]/(np.sum(predicted_abundance_map, axis=2) + 1e-6), vmin=0, vmax=1
        )
        ax[1, i].set_title(f"Accuracy: {accuracy_list[i]:.2f}, RMSE: {rmse_list[i]:.2f}")
        ax[0, i].axis("off")
        ax[1, i].axis("off")
    plt.savefig(os.path.join(args.save_folder, "abundance.png"))


def compute_abundance_rmse(
    abundance_map: np.ndarray, predicted_abundance_map: np.ndarray, good_permutation=None
):
    """
    Compute the RMSE for all permutation of the abundance maps

    Args:
        abundance_map (np.ndarray): _description_
        predicted_abundance_map (np.ndarray): _description_

    Returns:
        mse (float): _description_
    """

    if good_permutation is None:
        predicted_abundance_map = predicted_abundance_map/(np.sum(predicted_abundance_map, axis=2, keepdims=True) + 1e-6)
        rmse = 1e9
        rmse_list = [1e9] * 4
        permutation_endmember = list(permutations(range(4)))
        good_permutation = [0, 1, 2, 3]
        for permutation in permutation_endmember:
            rmse_permutation = np.sqrt(
                np.mean((abundance_map - predicted_abundance_map[:, :, permutation]) ** 2)
            )
            if rmse_permutation < rmse:
                rmse = rmse_permutation
                good_permutation = permutation
                rmse_list = np.sqrt(
                np.mean((abundance_map - predicted_abundance_map[:, :, permutation]) ** 2, axis=(0,1))
            )
    else:
        rmse = np.sqrt(
            np.mean((abundance_map - predicted_abundance_map[:, :, good_permutation]) ** 2)
        )
        rmse_list = np.sqrt(
            np.mean((abundance_map - predicted_abundance_map[:, :, good_permutation]) ** 2, axis=(0,1))
        )


    return rmse, good_permutation, rmse_list


def compute_accuracy(abundance_map, predicted_abundance_map, good_permutation=None):
    if good_permutation is None:
        permutation_endmember = list(permutations(range(abundance_map.shape[-1])))
        good_permutation = None

        binary_abundance_map = abundance_map > 0.5
        binary_predicted_abundance_map = predicted_abundance_map > 0.5

        accuracy = 0
        accuracy_list = np.zeros(len(permutation_endmember))
        x_range, y_range, nb_endmember = abundance_map.shape

        # Flatten the abundance maps for easier comparison
        binary_abundance_map_flat = binary_abundance_map.reshape(-1, nb_endmember)
        binary_predicted_abundance_map_flat = binary_predicted_abundance_map.reshape(
            -1, nb_endmember
        )

        for permutation in permutation_endmember:
            permuted_predicted = binary_predicted_abundance_map_flat[:, permutation]
            accuracy_permutation = np.mean(
                np.all(binary_abundance_map_flat == permuted_predicted, axis=1)
            )

            if accuracy_permutation > accuracy:
                accuracy = accuracy_permutation
                good_permutation = permutation
                accuracy_list = np.mean(binary_abundance_map_flat == permuted_predicted, axis=0)
    else:
        binary_abundance_map = abundance_map > 0.5
        binary_predicted_abundance_map = predicted_abundance_map > 0.5

        accuracy = np.mean(
            np.all(binary_abundance_map == binary_predicted_abundance_map[:, :, good_permutation], axis=2)
        )
        accuracy_list = np.mean(binary_abundance_map == binary_predicted_abundance_map[:, :, good_permutation], axis = (0, 1))
    return accuracy, good_permutation, accuracy_list


if __name__ == "__main__":
    main()
