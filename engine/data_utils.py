import random
import torch
from typing import Any, Dict, List, Tuple


def set_seed(seed: int = 1337):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, Any]]]):
    imgs, targets = list(zip(*batch))
    return list(imgs), list(targets)

