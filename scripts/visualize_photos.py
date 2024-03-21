from ipywidgets import interact
import numpy as np
import argparse
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt

from sentinel2_ts.utils.process_data import scale_data
from numpy.typing import ArrayLike


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Visualize photos")

    parser.add_argument(
        "--data_path", type=str, default=None, help="Path to the data to visualize"
    )
    parser.add_argument(
        "--show_data_training_zone", type=bool, default=False, help="Show the training zone in the data"
    )
    parser.add_argument(
        "--time_span",
        type=int,
        default=342,
        help="Number of time steps in the future predicted by the network",
    )
    parser.add_argument(
        "--initial_time",
        type=int,
        default=0,
        help="Initial time from which the visualization begin",
    )
    return parser


def main():
    args = create_arg_parser().parse_args()

    data = np.load(args.data_path)[
        args.initial_time : args.initial_time + args.time_span, [2, 1, 0], :, :
    ].transpose(0, 2, 3, 1)
    data = np.clip(scale_data(data, clipping=True) * 3, a_max=1, a_min=0)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)  # Adjust bottom to make room for the slider

    im = ax.imshow(data[args.initial_time])
    slider_ax = plt.axes(
        [0.25, 0.1, 0.65, 0.03], facecolor="lightgoldenrodyellow"
    )  # Define the slider's position and size
    slider = Slider(
        slider_ax,
        "Index",
        args.initial_time,
        args.initial_time + args.time_span - 1,
        valinit=args.initial_time,
        valstep=1,
    )  # Define the slider itself

    def update(val):
        image_index = slider.val
        im.set_data(data[int(image_index)])
        fig.canvas.draw_idle()  # Redraw the plot

    slider.on_changed(update)  # Call update when the slider value is changed
    
    if args.show_data_training_zone:
        draw_training_zone(ax, 60, 200, 100, 250)

    ax.axis("off")
    plt.show()

def draw_training_zone(ax, min_x, max_x, min_y, max_y):
    ax.plot(np.arange(min_x, max_x), min_y * np.ones(max_x - min_x), "r")
    ax.plot(np.arange(min_x, max_x), max_y * np.ones(max_x - min_x), "r")
    ax.plot(min_x * np.ones(max_y - min_y), np.arange(min_y, max_y), "r")
    ax.plot(max_x * np.ones(max_y - min_y), np.arange(min_y, max_y), "r")


if __name__ == "__main__":
    main()
