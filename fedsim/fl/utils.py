from functools import partial
from inspect import signature

import torch
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.convert_parameters import _check_param_device


def get_metric_scores(metric_fn_dict, y_true, y_pred):
    answer = {}
    if metric_fn_dict is None:
        return answer
    for name, fn in metric_fn_dict.items():
        args = dict()
        if "y_true" in signature(fn).parameters:
            args["y_true"] = y_true
        elif "target" in signature(fn).parameters:
            args["target"] = y_true
        else:
            raise NotImplementedError
        if "y_pred" in signature(fn).parameters:
            args["y_pred"] = y_pred
        elif "input" in signature(fn).parameters:
            args["input"] = y_pred
        else:
            raise NotImplementedError
        answer[name] = fn(**args)
    return answer


def default_closure(
    x,
    y,
    model,
    loss_fn,
    optimizer,
    metric_fn_dict,
    max_grad_norm=1000,
    link_fn=partial(torch.argmax, dim=1),
    device="cpu",
    transform_grads=None,
    transform_y=None,
    **kwargs,
):
    if transform_y is not None:
        y = transform_y(y)
    y_true = y.tolist()
    x = x.to(device)
    y = y.reshape(-1).long()
    y = y.to(device)
    model.train()
    outputs = model(x)
    loss = loss_fn(outputs, y)
    if loss.isnan() or loss.isinf():
        return loss
    # backpropagation
    loss.backward()
    if transform_grads is not None:
        transform_grads(model)
    # Clip gradients
    clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
    # optimize
    optimizer.step()
    optimizer.zero_grad()
    y_pred = link_fn(outputs).tolist()
    metrics = get_metric_scores(metric_fn_dict, y_true, y_pred)
    return loss, metrics


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


class ModelReconstructor(torch.nn.Module):
    def __init__(
        self, feature_extractor, classifier, connection_fn=None
    ) -> None:
        super(ModelReconstructor, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.connection_fn = connection_fn

    def forward(self, input):
        features = self.feature_extractor(input)
        if self.connection_fn is not None:
            features = self.connection_fn(features)
        return self.classifier(features)
