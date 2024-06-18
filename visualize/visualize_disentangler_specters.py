import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from tqdm import tqdm
from sentinel2_ts.dataset.process_data import scale_data, get_state_time_series
from sentinel2_ts.architectures import Disentangler
from sentinel2_ts.utils.visualize import plot_single_spectral_signature


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
        "--x", type=int, default=None, help="x value where to compute prediction"
    )
    parser.add_argument(
        "--y", type=int, default=None, help="y value where to compute prediction"
    )
    parser.add_argument(
        "--abundance_mode", type=str, default="conv", help="Abundance mode"
    )
    parser.add_argument("--endmember", type=str, default=None, help="Path to endmember")
    return parser


@torch.no_grad()
def main():
    args = create_argparser().parse_args()

    data = np.load(args.data_path)
    pixel_data = scale_data(data, clipping=args.clipping)[:, :, args.x, args.y]
    model = Disentangler(
        size=20,
        latent_dim=64,
        num_classes=args.num_classes,
        abundance_mode=args.abundance_mode,
    ).to(args.device)
    model.load_state_dict(torch.load(args.ckpt_path))
    model.eval()
    state = get_state_time_series(pixel_data, 1, 342).T.unsqueeze(0).to(args.device)
    disentangled_specters = (
        model.spectral_disentangler(state).cpu().detach().squeeze().numpy()
    )
    disentangled_specters = disentangled_specters.reshape(args.num_classes, 20, -1)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.subplots_adjust(bottom=0.25)  # Adjust bottom to make room for the time_slider

    plot_single_spectral_signature(ax, disentangled_specters[0, :, 0])

    time_slider_ax = plt.axes(
        [0.25, 0.1, 0.65, 0.03], facecolor="lightgoldenrodyellow"
    )  # Define the time_slider's position and siz
    time_slider = Slider(
        time_slider_ax, "Time", 0, 342 - 1, valinit=0, valstep=1
    )  # Define the time_slider itself

    class_slider_ax = plt.axes(
        [0.25, 0.15, 0.65, 0.03], facecolor="lightgoldenrodyellow"
    )
    class_slider = Slider(
        class_slider_ax, "Class", 0, args.num_classes - 1, valinit=0, valstep=1
    )

    def update(val):
        time = int(time_slider.val)
        class_idx = int(class_slider.val)
        ax.clear()
        plot_single_spectral_signature(ax, disentangled_specters[class_idx, :, time])
        fig.canvas.draw_idle()  # Redraw the plot

    time_slider.on_changed(update)
    class_slider.on_changed(update)
    ax.legend()
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Amplitude")
    plt.show()


if __name__ == "__main__":
    main()
