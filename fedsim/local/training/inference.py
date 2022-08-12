"""
Local Inference
---------------

Inference for local client
"""

import torch


def local_inference(
    model,
    data_loader,
    scores,
    device="cpu",
    transform_y=None,
):
    """to test the performance of a model on a test set.

    :param model: model to get the predictions from
    :param loader: data loader
    :param metric_fn_dict: a dict of {name: fn}, fn gets (inputs, targets)
    :param link_fn: to be applied on top of model output (e.g. softmax)
    :param device: device (e.g., 'cuda', '<gpu number> or 'cpu')
    """
    num_samples = 0
    model_is_training = model.training
    model.eval()
    with torch.no_grad():
        for (X, y) in data_loader:
            if transform_y is not None:
                y = transform_y(y)
            y = y.reshape(-1).long()
            y = y.to(device)
            X = X.to(device)
            outputs = model(X)
            num_samples += len(y)
            for score in scores.values():
                score(outputs, y)
            del outputs
    if model_is_training:
        model.train()
    return (num_samples,)
