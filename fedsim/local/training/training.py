"""
Local Training
--------------

Training for local client
"""
import inspect

from .step_closures import default_step_closure


def local_train(
    model,
    train_data_loader,
    epochs,
    steps,
    criterion,
    optimizer,
    lr_scheduler=None,
    device="cpu",
    step_closure=default_step_closure,
    scores=None,
    max_grad_norm=1000,
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
                loss = step_closure(
                    x,
                    y,
                    model,
                    criterion,
                    optimizer,
                    scores,
                    max_grad_norm,
                    device=device,
                    **step_ctx,
                )
                if loss.isnan() or loss.isinf():
                    del loss
                    diverged = True
                    break

                # update control variables
                epoch_step_cnt += 1
                num_steps += 1
                num_train_samples += y.shape[0]
                all_loss += loss.item()
                if lr_scheduler is not None:
                    step_args = inspect.signature(lr_scheduler.step).parameters
                    if "metrics" in step_args:
                        comb_scores = {**scores, **{criterion.get_name(): criterion}}
                        trigger_metric = lr_scheduler.trigger_metric
                        if trigger_metric not in comb_scores:
                            raise Exception(
                                f"{trigger_metric} not in local scores. "
                                f"Possible options are {comb_scores.keys()}"
                            )
                        lr_scheduler.step(comb_scores[trigger_metric].get_score())
                    else:
                        lr_scheduler.step()

    return (
        num_train_samples,
        num_steps,
        diverged,
    )
