from copy import deepcopy
from functools import partial
from typing import (
    Union, Iterator, Sized, Optional, Callable, AnyStr, Tuple, Dict, Iterable
    )

import torch
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.convert_parameters import _check_param_device
from tqdm import tqdm
from torch.nn import functional as F
import sys

from torch.utils.data import DataLoader
from collections import OrderedDict
import inspect


def inference(
    model, loader, metric_fn_dict, link_fn=partial(torch.argmax, dim=1), 
    device='cpu', loss_criterion=None
    ) -> Union[Dict, Tuple]:
    """

    :param model: model to get the predictions from
    :param loader: data loader
    :param metric_fn_dict: a dict of {name: fn},s.t. each fn gets (y_true, y_pred) as input and return a score
    :param link_fn: to be applied on top of model output (e.g. softmax) before passing to the metric scores
    :param device: device (e.g., <gpu number> or 'cpu') to perform heavy operations on
    :param loss_criterion: if specified, should be a function that takes the output of model and targets
    """
    y_true, y_pred = [], []
    loss = 0
    # num_batches = 0
    num_samples = 0
    model_is_training = model.training
    model.eval()
    with torch.no_grad():
        for (X, y) in loader:
            y_true.extend(y.tolist())
            y = y.reshape(-1).long()
            y = y.to(device)
            X = X.to(device)
            outputs = model(X)
            y_pred_batch = link_fn(outputs).tolist()
            y_pred.extend(y_pred_batch)
            # num_batches += 1
            num_samples += len(y)
            if loss_criterion is not None:
                loss += len(y) * loss_criterion(outputs, y).item()
            del outputs
    if model_is_training:
        model.train()
    if loss_criterion is None:
        return get_metric_scores(metric_fn_dict, y_true, y_pred)
    return get_metric_scores(
        metric_fn_dict, y_true, y_pred), loss / (float(num_samples)+1e-8
        )


def default_closure(x, y, model, loss_fn, optimizer, max_grad_norm=1000):
    loss = loss_fn(model(x), y)
    if loss.isnan() or loss.isinf():
        return loss
    # backpropagation
    loss.backward()
    clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)  # Clip gradients
    # optimize
    optimizer.step()
    optimizer.zero_grad()
    return loss


def local_train_val(
    model: torch.nn.Module, train_data_loader: Union[Iterator, Sized],
    val_data_loader: Optional[Union[Iterator, Sized]], epochs: int, steps: int,
    step_val_interval: int, checkpoint_metric: Callable, loss_fn: Callable, 
    optimizer: torch.optim.Optimizer, device: Union[AnyStr, int], 
    name: Optional[str] = None, step_closure: Optional[Callable] = None, 
    max_grad_norm: int = 1000) -> Tuple[
        Dict, int, int, int, float, bool, float
        ]:
    """

    :param model: model for training/validation
    :param train_data_loader: train data loader
    :param val_data_loader: valid data loader
    :param epochs: number of training epochs
    :param steps: number of extra steps to take after looping over the number of epochs specified in epochs argument
    :param step_val_interval: number of steps in between the validations, -1 means only validate after training
    :param loss_fn: loss function for training
    :param optimizer: optimizer
    :param device: an integer number indicating gpu number or 'cpu'
    :param checkpoint_metric: in case checkpoint_policy is 'best', this is the name of the metric to compare and save the best. It should be string that equals one of the metrics in metric_fn_list
    :param name: optional name for monitoring purpose
    :param step_closure: optional callable that takes loss, x, logits, y and optimizer and takes the step
    :param max_grad_norm: maximum norm of the gradients to clip
    :return:
    """

    if val_data_loader is not None and len(val_data_loader) > 0:
        validation_needed = True
    else:
        validation_needed = False

    if step_closure is None:
        step_closure = default_closure

    if steps > 0:
        # this is because we break out of the epoch loop, so we need an 
        # additional iteration to go over extra steps
        epochs += 1
    # instantiate control variables
    num_steps = 0

    best_val_score = 0.0
    model_checkpoint = None
    diverged = False
    
    total_loss = 0
    num_train_samples = 0
    if train_data_loader is not None:
        # iteration over epochs
        for epoch in range(epochs):
            epoch_step_cnt = 0
            model.train()
            if diverged:
                break
            # iteration over mini-batches
            for x, y in train_data_loader:
                # send the mini-batch to device
                x = x.to(device)
                y = y.reshape(-1).long()
                y = y.to(device)
                # calculate the local objective's loss
                loss = step_closure(
                    x, y, model, loss_fn, optimizer, max_grad_norm
                    )
                if loss.isnan() or loss.isinf():
                    del loss
                    diverged = True
                    break
                # update control variables
                epoch_step_cnt += 1
                num_steps += 1
                num_train_samples += y.shape[0]
                total_loss += loss.item()
    return num_train_samples, num_steps, diverged, total_loss

    #             if validation_needed and step_val_interval > 0 and \
    #                 num_steps % step_val_interval == 0:
    #                 val_scores = inference(
    #                     model=model, loader=val_data_loader,
    #                     metric_fn_dict={
    #                         'checkpoint_metric': checkpoint_metric
    #                         }, link_fn=partial(torch.argmax, dim=1), 
    #                         device=device
    #                         )
    #                 if val_scores['checkpoint_metric'] > best_val_score:
    #                     best_val_score = val_scores['checkpoint_metric']
    #                     model_checkpoint = deepcopy(model.state_dict())
    #                 if name is not None:
    #                     print('val scores of {}: {}'.format(name, val_scores))
    #             # check if last steps is taken
    #             if (epoch == epochs - 1) and (steps == epoch_step_cnt):
    #                 break

    # if model_checkpoint is None:
    #     model_checkpoint = model.state_dict()
    # num_val_samples = len(val_data_loader.dataset) if validation_needed else 0
    # model_checkpoint = deepcopy(model_checkpoint)
    # return model_checkpoint, num_train_samples, num_steps, num_val_samples, \
    #     best_val_score, diverged, total_loss


def get_metric_scores(metric_fn_dict, y_true, y_pred):
    answer = {}
    if metric_fn_dict is None:
        return answer
    for name, fn in metric_fn_dict.items():
        answer[name] = fn(y_true, y_pred)
    return answer


def vector_to_parameters_like(
    vec: torch.Tensor, parameters_like: Iterable[torch.Tensor]
    ) -> Iterable[torch.Tensor]:
    r"""Convert one vector to new parameters like the ones provided 

    Args:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model. This is only used to get the sizes. New 
            parametere are defined.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'
                        .format(torch.typename(vec)))
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
        new_params.append(vec[pointer:pointer + num_param].view_as(param).data)

        # Increment the pointer
        pointer += num_param
    return new_params
