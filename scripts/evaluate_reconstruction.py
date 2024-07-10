import os
import argparse

import numpy as np
import yaml
import matplotlib.pyplot as plt

from sentinel2_ts.utils.load_model import load_data


def create_argparser():
    parser = argparse.ArgumentParser(description="Evaluate Reconstruction")
    parser.add_argument("--data_path", type=str, default=None, help="Path to the data")
    parser.add_argument(
        "--reconstruction_path",
        type=str,
        default=None,
        help="Path to the reconstruction",
    )
    parser.add_argument(
        "--save_folder", type=str, default=None, help="Folder to save the results"
    )
    parser.add_argument("--scale_data", type=bool, help="Scale the data")
    parser.add_argument(
        "--clipping", type=bool, default=True, help="Whether to clip the data or not"
    )
    return parser


def main():
    args = create_argparser().parse_args()
    data = load_data(args)
    reconstruction = np.load(args.reconstruction_path)

    total_mse = np.mean((data - reconstruction) ** 2)

    print(f"Total MSE: {total_mse}")

    os.makedirs(args.save_folder, exist_ok=True)
    with open(os.path.join(args.save_folder, "reconstruction.yaml"), "w") as f:
        yaml.dump(
            {"MSE": float(total_mse)},
            f,
        )
    plt.figure()
    plt.imshow((np.mean((data - reconstruction) ** 2, axis=(0, 1))))
    plt.colorbar()
    plt.title("Reconstruction MSE")
    plt.axis("off")
    plt.savefig(os.path.join(args.save_folder, "reconstruction_mse.png"))
    prediction = reconstruction[:, 6, 67, 97]
    ground_truth = data[:, 6, 67, 97]
    plt.figure()
    plt.plot(prediction, label="Prediction")
    plt.plot(ground_truth, label="Ground Truth")
    plt.legend()
    plt.title("Reconstruction")
    plt.savefig(os.path.join(args.save_folder, "reconstruction_1.png"))
    prediction = reconstruction[:, 6, 167, 97]
    ground_truth = data[:, 6, 167, 97]
    plt.figure()
    plt.plot(prediction, label="Prediction")
    plt.plot(ground_truth, label="Ground Truth")
    plt.legend()
    plt.title("Reconstruction")
    plt.savefig(os.path.join(args.save_folder, "reconstruction_2.png"))



if __name__ == "__main__":
    main()
