from functools import partial

import torch
from torch.nn.utils import clip_grad_norm_

from fedsim.utils import collect_scores


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
    metrics = collect_scores(metric_fn_dict, y_true, y_pred)
    return loss, metrics
