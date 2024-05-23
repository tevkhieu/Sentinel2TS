import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from tqdm import tqdm
from sentinel2_ts.utils.load_model import koopman_model_from_ckpt
from sentinel2_ts.dataset.process_data import scale_data, get_state_all_data


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_matrix_k", help="Path to the matrix K file")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to the network checkpoint (only for koopman_ae)",
    )
    parser.add_argument("--data_path", type=str, default=None, help="Path to the data")
    parser.add_argument(
        "--clipping", type=bool, default=True, help="Clipping the data or not"
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
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

    data = np.load(args.data_path)
    data = scale_data(data, clipping=args.clipping)
    time_range, x_range, y_range = data.shape[0], data.shape[2], data.shape[3]
    abundance_map = np.zeros(
        (time_range - 1, x_range, y_range, args.latent_dim[-1]), dtype=np.float16
    )
    model = koopman_model_from_ckpt(
        args.ckpt_path, args.path_matrix_k, "koopman_unmixer", args.latent_dim
    ).to(args.device)
    model.eval()
    state_map = get_state_all_data(data)
    for t in tqdm(range(time_range - 1)):
        for x in range(x_range):
            abundance_map[t, x, :, :] = (
                model.get_abundance_remember(state_map[t, x, :, :].to(args.device), 2)
                .transpose(0, 1)
                .cpu()
                .detach()
                .numpy()
            )

    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.subplots_adjust(
        bottom=0.25
    )  # Adjust bottom to make room for the endmember_slider
    im = ax.imshow(abundance_map[0, :, :, 0], vmin=0, vmax=1, cmap="viridis")
    endmember_slider_ax = plt.axes(
        [0.25, 0.1, 0.65, 0.03], facecolor="lightgoldenrodyellow"
    )  # Define the endmember_slider's position and size
    endmember_slider = Slider(
        endmember_slider_ax,
        "endmember index",
        0,
        args.latent_dim[-1] - 1,
        valinit=0,
        valstep=1,
    )

    time_slider_ax = plt.axes(
        [0.25, 0.05, 0.65, 0.03], facecolor="lightgoldenrodyellow"
    )
    time_slider = Slider(
        time_slider_ax,
        "Time index",
        0,
        342 - 1,
        valinit=0,
        valstep=1,
    )

    def update(val):
        endmember_index = endmember_slider.val
        time_index = time_slider.val
        im.set_data(abundance_map[int(time_index), :, :, int(endmember_index)])
        fig.canvas.draw_idle()

    endmember_slider.on_changed(update)
    time_slider.on_changed(update)

    plt.colorbar(im)
    plt.show()


if __name__ == "__main__":
    main()
