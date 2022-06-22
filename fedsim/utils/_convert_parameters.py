import torch
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
        new_params.append(
            vec[pointer: pointer + num_param].view_as(param).data
        )

        # Increment the pointer
        pointer += num_param
    return new_params
