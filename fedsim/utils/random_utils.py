r"""
Random Utils
------------
"""
import random

import numpy as np
import torch


def set_seed(seed, use_cuda) -> None:
    """sets default random generator seed of ``numpy``, ``random`` and ``torch``.
    In case of using cuda, related randomness is also taken care of.

    Args:
        seed (_type_): _description_
        use_cuda (_type_): _description_
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)
