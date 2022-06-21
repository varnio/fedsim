from functools import partial

import torch

from fedsim.utils import add_dict_to_dict
from fedsim.utils import apply_on_dict

from .step_closures import default_closure


def local_train(
    model,
    train_data_loader,
    epochs,
    steps,
    loss_fn,
    optimizer,
    device,
    step_closure=default_closure,
    metric_fn_dict=None,
    max_grad_norm=1000,
    link_fn=partial(
        torch.argmax,
        dim=1,
    ),
    **step_ctx,
):

    if steps > 0:
        # this is because we break out of the epoch loop, so we need an
        # additional iteration to go over extra steps
        epochs += 1
    # instantiate control variables
    num_steps = 0

    diverged = False

    all_loss = 0
    num_train_samples = 0
    metrics = None

    if train_data_loader is not None:
        # iteration over epochs
        for _ in range(epochs):
            if diverged:
                break
            # iteration over mini-batches
            epoch_step_cnt = 0
            for x, y in train_data_loader:
                # send the mini-batch to device
                # calculate the local objective's loss
                loss, batch_metrics = step_closure(
                    x,
                    y,
                    model,
                    loss_fn,
                    optimizer,
                    metric_fn_dict,
                    max_grad_norm,
                    link_fn=link_fn,
                    device=device,
                    **step_ctx,
                )
                if loss.isnan() or loss.isinf():
                    del loss
                    diverged = True
                    break
                metrics = add_dict_to_dict(batch_metrics, metrics)

                # update control variables
                epoch_step_cnt += 1
                num_steps += 1
                num_train_samples += y.shape[0]
                all_loss += loss.item()

        # add average metrics over epochs
        normalized_metrics = apply_on_dict(
            metrics, lambda _, x: x / num_steps, return_as_dict=True
        )
        avg_loss = all_loss / num_steps

    return (
        num_train_samples,
        num_steps,
        diverged,
        avg_loss,
        normalized_metrics,
    )
