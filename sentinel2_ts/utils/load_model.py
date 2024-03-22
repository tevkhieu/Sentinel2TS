import torch
from sentinel2_ts.architectures.koopman_ae import KoopmanAE


def koopman_model_from_ckpt(ckpt_path: str, path_matrix_k: str):
    """
    Load Koopman model from checkpoint and matrix K.

    Args:
        ckpt_path (str): Path to the checkpoint file
        path_matrix_k (str): Path to the matrix K file

    Returns:
        _type_: _description_
    """
    model = KoopmanAE(20, [512, 256, 32])
    model.load_state_dict(torch.load(ckpt_path))
    model.K = torch.load(path_matrix_k)

    return model
