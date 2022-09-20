r"""
Parameters Conversion
---------------------
"""
from collections import OrderedDict

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.utils import parameters_to_vector
from torch.nn.utils import vector_to_parameters
from torch.nn.utils.convert_parameters import _check_param_device


def vector_to_parameters_like(vec, parameters_like):
    r"""Convert one vector to new parameters like the ones provided

    Args:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model. This is only used to get the sizes. New
            parametere are defined.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError(
            "expected torch.Tensor, but got: {}".format(torch.typename(vec))
        )
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    new_params = []
    for param in parameters_like:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the
        # parameter
        new_params.append(vec[pointer : pointer + num_param].view_as(param).data)

        # Increment the pointer
        pointer += num_param
    return new_params


def vector_to_named_parameters_like(
    vec: Tensor,
    named_parameters_like: OrderedDict[str, Tensor],
) -> OrderedDict[str, Tensor]:
    r"""Convert one vector to new named parameters like the ones provided

    Args:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (OrderedDict[str, Tensor]): a dictioanry of Tensors that are the
            parameters of a model. This is only used to get the sizes and keys. New
            parametere are defined.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError(
            "expected torch.Tensor, but got: {}".format(torch.typename(vec))
        )
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    new_params = OrderedDict()
    for key, param in named_parameters_like:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the
        # parameter
        new_params[key] = vec[pointer : pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param
    return new_params


def vectorize_module(module: Module, clone=True, detach=True):
    r"""convert parameters of a module to a vector

    Args:
        module (Module): module to convert the parameters of
        clone (bool, optional): clones the output. Defaults to True.
        detach (bool, optional): detaches the output. Defaults to True.

    Returns:
        Module: 1-D Tensor of all parameters in the module
    """
    vec = parameters_to_vector(module.parameters())
    if clone:
        vec = vec.clone()
    if detach:
        vec = vec.detach()
    return vec


def vectorize_module_grads(module: Module, clone=True, detach=True):
    r"""convert parameters gradients of a module to a vector

    Args:
        module (Module): module to convert the parameters of
        clone (bool, optional): clones the output. Defaults to True.
        detach (bool, optional): detaches the output. Defaults to True.

    Returns:
        Module: 1-D Tensor of gradients of all parameters in the module. None if at
            least grad of one children deos not exist.
    """

    param_device = None

    vec = []
    for param in module.parameters():
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        if param.grad is None:
            return None
        else:
            vec.append(param.grad.view(-1))
    vec = torch.cat(vec)
    if clone:
        vec = vec.clone()
    if detach:
        vec = vec.detach()
    return vec


def initialize_module(module: Module, vec: Tensor, clone=True, detach=True):
    r"""initializes a module's parameters with a 1-D vector

    Args:
        module (Module): module to initialize weights
        vec (Tensor): a 1-D Tensor
        clone (bool, optional): clones the vector before initilization.
            Defaults to True.
        detach (bool, optional): detaches the output before the initialization.
            Defaults to True.
    """
    if clone:
        vec = vec.clone()
    if detach:
        vec = vec.detach()
    if len(vectorize_module(module)) != len(vec):
        return False

    vector_to_parameters(vec, module.parameters())
    return True
