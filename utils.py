import importlib
import numpy as np
import torch
import random


def get_from_module(module_name, submodule_name, class_name='Algorithm'):
    module = importlib.import_module(module_name)
    if submodule_name in module.__all__:
        sub_module = importlib.import_module(
            '{}.{}'.format(module_name, submodule_name)
            )
        return getattr(sub_module, class_name)
    raise NotImplementedError


def set_seed(seed, use_cuda):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    if use_cuda:
        torch.cuda.manual_seed_all(seed)