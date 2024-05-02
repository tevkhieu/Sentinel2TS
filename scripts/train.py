import os
import argparse
import lightning as L
from sentinel2_ts.runners import LitKoopmanAE, LitKoopmanUnmixer, LitLinear, LitLSTM, LitDisentangler
from sentinel2_ts.utils.get_dataloader import get_dataloader


def create_arg_parser():
    parser = argparse.ArgumentParser(
        description="General script for training priors or downstream tasks"
    )

    parser.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment"
    )
    parser.add_argument(
        "--train_data_path", type=str, default=None, help="Path to the training data"
    )
    parser.add_argument(
        "--val_data_path", type=str, default=None, help="Path to the validation data"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--time_span",
        type=int,
        default=100,
        help="Number of time steps in the future predicted by the network",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="lstm",
        help="lstm | linear | koopman_ae | unmixer chooses which architecture to use",
    )
    parser.add_argument(
        "--max_epochs", type=int, default=150, help="Max number of epochs"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device used")
    parser.add_argument(
        "--minimum_x", type=int, default=60, help="Minimal x value on the image"
    )
    parser.add_argument(
        "--maximum_x", type=int, default=200, help="Maximal x value on the image"
    )
    parser.add_argument(
        "--minimum_y", type=int, default=100, help="Minimal y value on the image"
    )
    parser.add_argument(
        "--maximum_y", type=int, default=250, help="Maximal y value on the image"
    )
    parser.add_argument("--size", type=int, default=20, help="Size of the model")
    parser.add_argument(
        "--latent_dim",
        type=int,
        nargs="+",
        default=[512, 256, 32],
        help="Latent dimension",
    )
    parser.add_argument("--data_mode", type=str, default=None, help="Data mode")
    return parser


def main():
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()

    train_dataloader = get_dataloader(
        args.train_data_path,
        data_mode=args.data_mode,
        batch_size=args.batch_size,
        time_span=args.time_span,
        dataset_len=512 * 512,
        minimal_x=args.minimum_x,
        maximal_x=args.maximum_x,
        minimal_y=args.minimum_y,
        maximal_y=args.maximum_y,
    )
    val_dataloader = get_dataloader(
        args.val_data_path,
        data_mode=args.data_mode,
        dataset_len=512 * 51,
        batch_size=args.batch_size,
        time_span=args.time_span,
        shuffle=False,
        num_workers=2,
        minimal_x=250,
        maximal_x=400,
        minimal_y=250,
        maximal_y=400,
    )

    trainer = L.Trainer(
        accelerator=args.device,
        max_epochs=args.max_epochs,
        default_root_dir=os.path.join(
            os.getcwd(), "models_lightning_ckpt", args.experiment_name
        ),
    )

    match args.mode:
        case "lstm":
            model = LitLSTM(
                expermiment_name=args.experiment_name, time_span=args.time_span
            )
        case "linear":
            model = LitLinear(size=args.size, experiment_name=args.experiment_name)
        case "koopman_ae":
            model = LitKoopmanAE(
                size=args.size,
                experiment_name=args.experiment_name,
                time_span=args.time_span,
                device=args.device,
            )
        case "unmixer":
            model = LitKoopmanUnmixer(
                size=args.size,
                experiment_name=args.experiment_name,
                latent_dim=args.latent_dim,
                time_span=args.time_span,
                device=args.device,
            )
        case "spectral_disentangler":
            model = LitDisentangler(
                size=args.size,
                num_classes=10,
                experiment_name=args.experiment_name,
            )

    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )


if __name__ == "__main__":
    main()
