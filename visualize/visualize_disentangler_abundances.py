import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from tqdm import tqdm
from sentinel2_ts.dataset.process_data import scale_data, get_state_all_data
from sentinel2_ts.architectures import Disentangler


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", help="Path to the abundance disentangler model")
    parser.add_argument("--data_path", type=str, default=None, help="Path to the data")
    parser.add_argument(
        "--clipping", type=bool, default=True, help="Clipping the data or not"
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of classes")
    parser.add_argument(
        "--abundance_mode", type=str, default="conv", help="Abundance mode"
    )
    return parser


@torch.no_grad()
def main():
    args = create_argparser().parse_args()

    data = np.load(args.data_path)
    data = scale_data(data, clipping=args.clipping)
    time_range, x_range, y_range = data.shape[0], data.shape[2], data.shape[3]
    abundance_map = np.zeros((x_range, y_range, args.num_classes), dtype=np.float16)
    total_map = np.zeros((x_range, y_range), dtype=np.float16)
    model = Disentangler(
        size=20,
        latent_dim=64,
        num_classes=args.num_classes,
        abundance_mode=args.abundance_mode,
    ).to(args.device)
    model.load_state_dict(torch.load(args.ckpt_path))
    model.eval()
    state_map = (
        get_state_all_data(data).transpose(0, -1).transpose(0, -2).transpose(0, -3)
    )
    for x in tqdm(range(x_range)):
        abundance_map[x, :, :] = (
            model.abundance_disentangler(state_map[x, :, :].to(args.device))
            .cpu()
            .detach()
            .numpy()
        )

    # Normalize if necessary
    if np.max(abundance_map) > 1e6:  # Example threshold
        abundance_map /= np.max(abundance_map)

    # Safe division
    with np.errstate(over="ignore", divide="ignore"):
        for x in range(abundance_map.shape[0]):
            for y in range(abundance_map.shape[1]):
                total = np.sum(abundance_map[x, y, :]) + 1e-6
                total_map[x, y] = total
                if not np.isinf(total) and not np.isnan(total):
                    abundance_map[x, y, :] /= total

    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.subplots_adjust(
        bottom=0.25
    )  # Adjust bottom to make room for the endmember_slider
    im = ax.imshow(abundance_map[:, :, 0], vmin=0, vmax=1, cmap="viridis")
    endmember_slider_ax = plt.axes(
        [0.25, 0.1, 0.65, 0.03], facecolor="lightgoldenrodyellow"
    )  # Define the endmember_slider's position and size
    endmember_slider = Slider(
        endmember_slider_ax,
        "endmember index",
        0,
        args.num_classes - 1,
        valinit=0,
        valstep=1,
    )

    def update(val):
        endmember_index = endmember_slider.val
        im.set_data(abundance_map[:, :, int(endmember_index)])
        fig.canvas.draw_idle()

    endmember_slider.on_changed(update)

    plt.colorbar(im)
    plt.show()

    plt.imshow(total_map)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
