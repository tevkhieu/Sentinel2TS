import torch
import argparse

from sentinel2_ts.runners.lit_koopman_ae import LitKoopmanAE
from sentinel2_ts.runners.lit_linear import LitLinear
from sentinel2_ts.runners.lit_lstm import LitLSTM


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="Path to the checkpoint file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path where the state dict will be saved",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="koopman_ae | linear | lstm chooses which architecture to use",
    )

    return parser


def main():
    args = create_arg_parser().parse_args()

    # Load the checkpoint
    checkpoint = torch.load(args.ckpt_path)

    # Get the state dict of the model from the checkpoint
    model_state_dict = checkpoint["state_dict"]
    new_state_dict = {}

    for key, item in model_state_dict.items():
        new_key = ".".join(key.split("."))[1:][1:]
        new_state_dict[new_key] = item

    # Save the model state dict to a file
    torch.save(new_state_dict, args.output_path)


if __name__ == "__main__":
    main()
