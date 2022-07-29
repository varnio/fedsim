import random

import numpy as np
import torch


def set_seed(seed, use_cuda) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)
