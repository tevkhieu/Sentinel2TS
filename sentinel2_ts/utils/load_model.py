import numpy as np
import torch
from sentinel2_ts.architectures import (
    Linear,
    LSTM,
    Disentangler,
    KoopmanAE,
    KoopmanUnmixer,
)
from sentinel2_ts.dataset.process_data import scale_data


def koopman_model_from_ckpt(
    size: int,
    ckpt_path: str,
    path_matrix_k: str,
    double_decoder: bool = True,
    mode: str = "koopman_ae",
    latent_dim: list[int] = [512, 256, 32],
):
    """
    Load Koopman model from checkpoint and matrix K.

    Args:
        ckpt_path (str): Path to the checkpoint file
        path_matrix_k (str): Path to the matrix K file

    Returns:
        _type_: _description_
    """
    if mode == "koopman_ae":
        model = KoopmanAE(size, latent_dim)
    else:
        model = KoopmanUnmixer(size, latent_dim, double_decoder=double_decoder)
    model.load_state_dict(torch.load(ckpt_path))
    model.K = torch.load(path_matrix_k)

    return model


def load_model(args):
    match args.mode:
        case "lstm":
            model = LSTM(args.size, 256, 20)
            model.load_state_dict(torch.load(args.ckpt_path))
        case "linear":
            model = Linear(args.size)
            model.load_state_dict(torch.load(args.ckpt_path))
        case "koopman_ae":
            model = koopman_model_from_ckpt(
                args.size,
                args.ckpt_path,
                args.path_matrix_k,
                "koopman_ae",
                args.latent_dim,
            )
        case "koopman_unmixer":
            model = koopman_model_from_ckpt(
                args.size,
                args.ckpt_path,
                args.path_matrix_k,
                args.double_decoder,
                "koopman_unmixer",
                args.latent_dim,
            )
        case "disentangler":
            model = Disentangler(
                size=args.size, latent_dim=64, num_classes=args.num_classes, abundance_mode=args.abundance_mode, disentangler_mode=args.disentangler_mode
            )
            state_dict_spectral_disentangler = torch.load(args.ckpt_path)
            model.load_state_dict(state_dict_spectral_disentangler)
        case _:
            raise ValueError("Mode not recognized")
    return model


def load_data(args):
    data = np.load(args.data_path)
    if args.scale_data:
        data = scale_data(data, clipping=args.clipping)
    return data
