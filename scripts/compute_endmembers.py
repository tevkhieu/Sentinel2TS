import os
import argparse
import numpy as np

import torch
from sentinel2_ts.dataset.process_data import get_state_time_series
from sentinel2_ts.architectures import Disentangler, KoopmanUnmixer


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", help="Path to the abundance disentangler model")
    parser.add_argument(
        "--dataset_path", type=str, default=None, help="Path to the data"
    )
    parser.add_argument(
        "--clipping", type=bool, default=True, help="Clipping the data or not"
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of classes")
    parser.add_argument(
        "--x", type=int, default=None, help="x value where to compute prediction"
    )
    parser.add_argument(
        "--y", type=int, default=None, help="y value where to compute prediction"
    )
    parser.add_argument(
        "--abundance_mode", type=str, default="conv", help="Abundance mode"
    )
    parser.add_argument("--endmember", type=str, default=None, help="Path to endmember")
    parser.add_argument(
        "--save_folder", type=str, default=None, help="Folder for results"
    )
    parser.add_argument("--size", type=int, default=20, help="Size of the data")
    parser.add_argument("--mode", type=str, default=None, help="Mode of operation")
    parser.add_argument(
        "--latent_dim",
        type=int,
        nargs="+",
        default=[512, 256, 32],
        help="Latent dimension",
    )
    return parser


@torch.no_grad()
def main():
    args = create_argparser().parse_args()

    if args.mode == "disentangler":
        pixel_data = np.load(
            os.path.join(args.dataset_path, f"{args.x:03}_{args.y:03}.npy")
        )
        model = Disentangler(
            size=args.size,
            latent_dim=64,
            num_classes=args.num_classes,
            abundance_mode=args.abundance_mode,
        ).to(args.device)
        model.load_state_dict(torch.load(args.ckpt_path))
        model.eval()
        state = get_state_time_series(pixel_data, 1, 342).T.unsqueeze(0).to(args.device)
        endmembers = model.spectral_disentangler(state).cpu().detach().squeeze().numpy()
        endmembers = endmembers.reshape(args.num_classes, args.size, -1)

    elif args.mode == "unmixer":
        model = KoopmanUnmixer(args.size, args.latent_dim)
        model.load_state_dict(torch.load(args.ckpt_path))

        endmembers = (
            model.final_layer.weight[: args.size // 2, :]
            .cpu()
            .detach()
            .numpy()
            .transpose(1, 0)
        )

    os.makedirs(args.save_folder, exist_ok=True)
    np.save(os.path.join(args.save_folder, "endmembers.npy"), endmembers)


if __name__ == "__main__":
    main()
