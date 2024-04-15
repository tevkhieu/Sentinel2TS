import torch
from sentinel2_ts.architectures import KoopmanAE, KoopmanUnmixer


def koopman_model_from_ckpt(
    ckpt_path: str,
    path_matrix_k: str,
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
        model = KoopmanAE(20, latent_dim)
    else:
        model = KoopmanUnmixer(20, latent_dim)
    model.load_state_dict(torch.load(ckpt_path))
    model.K = torch.load(path_matrix_k)

    return model
