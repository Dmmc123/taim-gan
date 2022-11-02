import torch
from pathlib import Path
from typing import Any

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

config_path = Path(__file__).parent.parent.absolute()
output_path = config_path / "models"

print(type(output_path))

output_path = output_path  / "generated" / "model.pth"
print(output_path)


config_dict = {
    'Ng': 32,
    'D': 256,
    'condition_dim': 100,
    'noise_dim': 100,
    'disc_lr': 2e-4,
    'gen_lr': 2e-4,
    'batch_size': 64,
    'device': device,
    'epochs':  200,
    'output_dir': output_path,
    'snapshot': 15,
    'const_dict': {
        'smooth_val_gen': 0.999,
        'lambda1': 1,
        'lambda2': 1,
        'lambda3': 1,
        'lambda4': 1,
        'gamma1': 4,
        'gamma2': 5,
        'gamma3': 10,
    }
}

def update_config(config_dict: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    for key, value in kwargs.items():
        config_dict[key] = value
    return config_dict