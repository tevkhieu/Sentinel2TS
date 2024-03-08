import sys
import os
import argparse
import numpy as np
import torch

# sys.path.append("..")

from sentinel2_ts.runners.data_assimilation import DataAssimilation
from sentinel2_ts.utils.process_data import scale_data
from sentinel2_ts.architectures.lstm import LSTM



def create_arg_parser():
    parser = argparse.ArgumentParser(description="General script for training priors or downstream tasks")

    parser.add_argument("--experiment_name", type=str, default=None, help="Name of the experiment")
    parser.add_argument("--time_span", type=int, default=100, help="Number of time steps in the future predicted by the network")
    parser.add_argument("--mode", type=str, default="lstm", help="lstm | linear | koopman_ae chooses which architecture to use")
    parser.add_argument("--max_epochs", type=int, default=200, help="Max number of epochs")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to the network checkpoint")
    parser.add_argument("--data_path", type=str, default=None, help="Path to the data where assimilation will be done")
    parser.add_argument("--min_x", type=int, default=0, help="Minimal value of x where data will be assimilated")
    parser.add_argument("--max_x", type=int, default=100, help="Maximal value of x where data will be assimilated")
    parser.add_argument("--min_y", type=int, default=0, help="Minimal value of y where data will be assimilated")
    parser.add_argument("--max_y", type=int, default=100, help="Maximal value of y where data will be assimilated")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for data assimilation")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for training")
    parser.add_argument("--clipping", type=bool, default=True, help="Whether to clip the data to 1 or not")

    return parser

def main():
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()

    log_dir_path = os.path.join("data_assimilation", "logs", args.experiment_name)
    output_path = os.path.join("data_assimilation", "assimilated_data", args.experiment_name,"assimilated_states.npy")
    
    os.makedirs(log_dir_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    data_assimilation = DataAssimilation(time_span=args.time_span, lr=args.lr, nb_epochs=args.max_epochs, device=args.device, log_dir=log_dir_path)

    if args.mode == "lstm":
        model = LSTM(20, 256, 20).to(args.device)

    state_dict = torch.load(args.ckpt_path)
    model.load_state_dict(state_dict)

    data = scale_data(np.load(args.data_path), clipping=args.clipping)
    assimilated_states = data_assimilation.data_assimilation(model, data)

    np.save(output_path, assimilated_states)

if __name__ == "__main__":
    main()

