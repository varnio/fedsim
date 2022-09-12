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
    """one step of local training including:
    * prepare mini batch of the data
    * forward pass
    * loss calculation
    * backward pass
    * transfor and modify the gradients
    * take optimization step
    * evaluate scores on the training mini-batch batch.

    Args:
        x (Tensor): inputs
        y (Tensor): labels
        model (Module): model
        criterion (Callable): loss criterion
        optimizer (Optimizer): optimizer chosen and instanciated from classes under
            ``torch.optim``.
        scores: Dict[str, Score]: dictionary of form str: Score to evaluate at the end
            of the closure.
        max_grad_norm (int, optional): to clip the norm of the gradients.
            Defaults to 1000.
        device (str, optional): device to load the data into
            ("cpu", "cuda", or device ordinal number). This must be the same device as
            the one model parameters are loaded into. Defaults to "cpu".
        transform_grads (Callable, optional): A function the takes the model and
            modified the gradients of the parameters. Defaults to None.
        transform_y (Callable, optional): a function that takes raw labels and modifies
            them. Defaults to None.

    Returns:
        Tensor: loss value obtained from the forward pass.
    """
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

    if scores is not None:
        for score in scores.values():
            score(outputs, y)
    return loss
