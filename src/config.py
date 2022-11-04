"""Configurations for the project."""
from pathlib import Path
from typing import Any

import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

repo_path = Path(__file__).parent.parent.absolute()
output_path = repo_path / "models"

config_dict = {
    "Ng": 32,
    "D": 256,
    "condition_dim": 100,
    "noise_dim": 100,
    "disc_lr": 2e-4,
    "gen_lr": 2e-4,
    "batch_size": 64,
    "device": device,
    "epochs": 200,
    "output_dir": output_path,
    "snapshot": 5,
    "const_dict": {
        "smooth_val_gen": 0.999,
        "lambda1": 1,
        "lambda2": 1,
        "lambda3": 1,
        "lambda4": 1,
        "gamma1": 4,
        "gamma2": 5,
        "gamma3": 10,
    },
}


def update_config(cfg_dict: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """
    Function to update the configuration dictionary.
    """
    for key, value in kwargs.items():
        cfg_dict[key] = value
    return cfg_dict
