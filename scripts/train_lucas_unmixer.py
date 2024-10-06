import argparse

from sentinel2_ts.runners import TrainerLucasUnmixer


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Lucas Unmixer training script")

    parser.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment"
    )
    parser.add_argument(
        "--image_dataset_path", type=str, default=None, help="Path to the image dataset"
    )
    parser.add_argument(
        "--time_series_dataset_path",
        type=str,
        default=None,
        help="Path to the time series dataset",
    )
    parser.add_argument("--num_classes", type=int, default=3, help="Number of classes")
    parser.add_argument("--size", type=int, default=20, help="Size of the model")
    parser.add_argument(
        "--latent_dim", type=int, nargs="+", help="Dimension of the latent space"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--max_epochs", type=int, default=150, help="Max number of epochs"
    )
    parser.add_argument("--max_x", type=int, default=199, help="Max x value")
    parser.add_argument("--min_y", type=int, default=0, help="Min y value")
    parser.add_argument("--max_y", type=int, default=199, help="Max y value")
    parser.add_argument("--min_x", type=int, default=0, help="Min x value")
    parser.add_argument(
        "--endmembers_path",
        type=str,
        default=None,
        help="Path to the endmembers file",
    )
    return parser


def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    trainer = TrainerLucasUnmixer(
        num_classes=args.num_classes,
        experiment_name=args.experiment_name,
        size=args.size,
        latent_dim=args.latent_dim,
        image_dataset_path=args.image_dataset_path,
        time_series_dataset_path=args.time_series_dataset_path,
        lr=args.lr,
        batch_size=args.batch_size,
        maximal_x=args.max_x,
        minimal_y=args.min_y,
        maximal_y=args.max_y,
        minimal_x=args.min_x,
        endmembers_path=args.endmembers_path,
    )
    trainer.train(max_epochs=args.max_epochs)


if __name__ == "__main__":
    main()
