import os
import argparse
import numpy as np
from sentinel2_ts.runners import DatasetGenerator


def create_argparse():
    parser = argparse.ArgumentParser(description="Extract the specter from the file")
    parser.add_argument("filename", type=str, help="name of the specter")
    parser.add_argument("file_path", type=str, help="path to the file")
    parser.add_argument(
        "save_folder", type=str, help="path to the folder to save the extracted specter"
    )
    return parser


def main():
    args = create_argparse().parse_args()
    specter_extractor = DatasetGenerator()
    os.makedirs(args.save_folder, exist_ok=True)

    specter = specter_extractor.extract_specter(args.file_path)
    save_path = os.path.join(args.save_folder, args.filename + ".npy")
    np.save(save_path, specter)


if __name__ == "__main__":
    main()
