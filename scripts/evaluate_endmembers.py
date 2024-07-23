import argparse
import os
from itertools import permutations
import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml


def create_parser():
    parser = argparse.ArgumentParser(description="Script for evaluating abundance")

    parser.add_argument(
        "--endmembers_path", type=str, default=None, help="Path to abundance map"
    )
    parser.add_argument(
        "--predicted_endmembers_path", type=str, default=None, help="Path to results"
    )
    parser.add_argument("--save_folder", type=str, default=None, help="Folder to save")
    parser.add_argument("--mode", type=str, default=None, help="Mode of operation")
    parser.add_argument(
        "--good_permutation", type=int, nargs="+", default=None, help="Good permutation"
    )
    return parser


@torch.no_grad()
def main():
    args = create_parser().parse_args()

    endmembers = np.load(args.endmembers_path)[1:]
    predicted_endmembers = np.load(args.predicted_endmembers_path)
    match args.mode:
        case "disentangler":
            predicted_endmembers = predicted_endmembers[
                :, : predicted_endmembers.shape[1] // 2, 0
            ]
        case "dynamical_unmixing":
            predicted_endmembers = predicted_endmembers[0].transpose(1, 0)
        case "cpd":
            predicted_endmembers = predicted_endmembers.transpose(1, 0)
        case "koopman_unmixer":
            pass
        case _:
            raise ValueError("Mode not supported")

    angle, good_permutation, angle_list = _compute_endmember_angle(
        endmembers, predicted_endmembers, args.good_permutation
    )

    print(f"Angle: {angle}, Good Permutation: {good_permutation}")

    # Save the results
    os.makedirs(args.save_folder, exist_ok=True)
    with open(os.path.join(args.save_folder, "endmembers.yaml"), "w") as f:
        yaml.dump(
            {
                "Angle": float(angle),
                "Good Permutation": list(good_permutation),
                "Angle List": [float(c) for c in angle_list],
            },
            f,
        )

    fig, ax = plt.subplots(1, 4, figsize=(30, 10))
    for i in range(endmembers.shape[0]):
        ax[i].plot(endmembers[i], label="Ground Truth")
        ax[i].plot(predicted_endmembers[good_permutation[i]], label="Predicted")
        ax[i].set_title(f"Angle: {angle_list[i]:.2f}")
    plt.legend()
    plt.savefig(os.path.join(args.save_folder, "endmembers.png"))


def _compute_endmember_angle(
    endmembers: np.ndarray,
    predicted_endmembers: np.ndarray,
    good_permutation: list[int] = None,
):
    """
    Compute the angle for all permutation of the endmembers

    Args:
        endmembers (np.ndarray): _description_
        predicted_endmembers (np.ndarray): _description_

    Returns:
        cps (float): _description_
    """
    angle = 1e9
    if good_permutation is None:
        permutation_endmember = list(permutations(range(4)))
        good_permutation = None
        angle_list_permutation = None
        for permutation in permutation_endmember:
            angle_permutation, angle_list_permutation = _compute_angle_list(
                endmembers, predicted_endmembers, permutation
            )
            if angle_permutation < angle:
                angle = angle_permutation
                good_permutation = permutation
                angle_list = angle_list_permutation
    else:
        angle, angle_list = _compute_angle_list(
            endmembers, predicted_endmembers, good_permutation
        )
    return angle, good_permutation, angle_list


def _compute_angle_list(
    endmembers: np.ndarray,
    predicted_endmembers: np.ndarray,
    permutation: list[int] = None,
):
    """
    Compute the angle for all permutation of the endmembers

    Args:
        endmembers (np.ndarray): _description_
        predicted_endmembers (np.ndarray): _description_

    Returns:
        cps (float): _description_
    """
    angle_list = (
        np.arccos(
            [
                np.dot(endmembers[i], predicted_endmembers[permutation[i]])
                / (
                    np.linalg.norm(endmembers[i])
                    * np.linalg.norm(predicted_endmembers[permutation[i]])
                )
                for i in range(endmembers.shape[0])
            ]
        )
        * 180
        / np.pi
    )

    angle = np.mean(angle_list)

    return angle, angle_list


if __name__ == "__main__":
    main()
