import os
import numpy as np
import torch

def label2onehot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    out = torch.zeros(labels.size(0), num_classes, dtype=torch.float32, device=labels.device)
    out[torch.arange(labels.size(0)), labels.long()] = 1.0
    return out

def seed_torch(seed: int = 10):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
