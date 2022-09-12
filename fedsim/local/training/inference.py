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

    Args:
        model (Module): model to get the predictions from
        data_loader (Iterable): inference data loader.
        scores (Dict[str, Score]): scores to evaluate
        device (str, optional): device to load the data into
            ("cpu", "cuda", or device ordinal number). This must be the same device as
            the one model parameters are loaded into. Defaults to "cpu".
        transform_y (Callable, optional): a function that takes raw labels and modifies
            them. Defaults to None.

    Returns:
        int: number of samples the evaluation is done for.
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
    return num_samples
