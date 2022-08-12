"""
Step Closures
-------------
"""

from torch.nn.utils import clip_grad_norm_


def default_step_closure(
    x,
    y,
    model,
    criterion,
    optimizer,
    scores,
    max_grad_norm=1000,
    device="cpu",
    transform_grads=None,
    transform_y=None,
    **kwargs,
):
    if transform_y is not None:
        y = transform_y(y)
    x = x.to(device)
    y = y.reshape(-1).long()
    y = y.to(device)
    model.train()
    outputs = model(x)
    loss = criterion(outputs, y)
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
    for score in scores.values():
        score(outputs, y)
    return loss
