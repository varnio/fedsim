from functools import partial

import torch

from fedsim.utils import collect_scores


def local_inference(
    model,
    data_loader,
    metric_fn_dict,
    link_fn=partial(torch.argmax, dim=1),
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
    y_true, y_pred = [], []
    num_samples = 0
    model_is_training = model.training
    model.eval()
    with torch.no_grad():
        for (X, y) in data_loader:
            if transform_y is not None:
                y = transform_y(y)
            y_true.extend(y.tolist())
            y = y.reshape(-1).long()
            y = y.to(device)
            X = X.to(device)
            outputs = model(X)
            y_pred_batch = link_fn(outputs).tolist()
            y_pred.extend(y_pred_batch)
            num_samples += len(y)
            del outputs
    if model_is_training:
        model.train()
    return (
        collect_scores(metric_fn_dict, y_true, y_pred),
        num_samples,
    )
