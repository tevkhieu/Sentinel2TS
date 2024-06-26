import argparse
import numpy as np
from sentinel2_ts.runners import ResidualSpecter


def create_argparse():
    parser = argparse.ArgumentParser(description="Train Residual Specter")
    parser.add_argument("--data", type=str, required=True, help="Path to the data")
    parser.add_argument(
        "--endmembers", type=str, required=True, help="Path to the endmembers"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    return parser


def main():
    parser = create_argparse()
    args = parser.parse_args()

    data = np.load(args.data)
    model = ResidualSpecter(
        data, np.load(args.endmembers), device=args.device, lr=args.lr
    )
    model.train()


if __name__ == "__main__":
    main()
