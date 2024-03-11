import os
import argparse
import lightning as L
from sentinel2_ts.runners.lit_lstm import LitLSTM
from sentinel2_ts.runners.lit_linear import LitLinear
from sentinel2_ts.utils.get_dataloader import get_dataloader

def create_arg_parser():
    parser = argparse.ArgumentParser(description="General script for training priors or downstream tasks")

    parser.add_argument("--experiment_name", type=str, default=None, help="Name of the experiment")
    parser.add_argument("--train_data_path", type=str, default=None, help="Path to the training data")
    parser.add_argument("--val_data_path", type=str, default=None, help="Path to the validation data")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--time_span", type=int, default=100, help="Number of time steps in the future predicted by the network"
    )
    parser.add_argument(
        "--mode", type=str, default="lstm", help="lstm | linear | koopman_ae chooses which architecture to use"
    )
    parser.add_argument("--max_epochs", type=int, default=500, help="Max number of epochs")
    parser.add_argument("--device", type=str, default="cuda", help="Device used")

    return parser


def main():
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()

    train_dataloader = get_dataloader(
        args.train_data_path, batch_size=args.batch_size, time_span=args.time_span, dataset_len=512 * 512
    )
    val_dataloader = get_dataloader(
        args.val_data_path,
        dataset_len=2600,
        batch_size=args.batch_size,
        time_span=args.time_span,
        shuffle=False,
        num_workers=2,
    )

    trainer = L.Trainer(
        accelerator=args.device,
        max_epochs=args.max_epochs,
        default_root_dir=os.path.join(os.getcwd(), "models_lightning_ckpt", args.experiment_name),
    )

    if args.mode == "lstm":
        model = LitLSTM(expermiment_name=args.experiment_name, time_span=args.time_span)

    elif args.mode == "linear":
        model = LitLinear(size=20, experiment_name=args.experiment_name)

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    main()
