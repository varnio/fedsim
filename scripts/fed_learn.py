r"""
fed-learn cli Command
---------------------
"""

import logging
import os
from functools import partial
from pprint import pformat
from typing import Optional

import click
import torch
from logall import TensorboardLogger

from fedsim import scores
from fedsim.utils import set_seed

from .utils import OptionEatAll
from .utils import decode_margs
from .utils import get_definition


@click.command(
    name="fed-learn",
    help="Simulates a Federated Learning system.",
)
@click.option(
    "--rounds",
    "-r",
    type=int,
    default=100,
    show_default=True,
    help="number of communication rounds.",
)
@click.option(
    "--data-manager",
    "-d",
    type=tuple,
    cls=OptionEatAll,
    show_default=True,
    default="BasicDataManager",
    help="name of data manager.",
)
@click.option(
    "--num-clients",
    "-n",
    type=int,
    default=500,
    show_default=True,
    help="number of clients.",
)
@click.option(
    "--client-sample-scheme",
    type=str,
    default="uniform",
    show_default=True,
    help="client sampling scheme (uniform or sequential for now).",
)
@click.option(
    "--client-sample-rate",
    "-c",
    type=float,
    default=0.01,
    show_default=True,
    help="mean portion of num clients to sample.",
)
@click.option(
    "--dataset-root",
    type=str,
    default="data",
    show_default=True,
    help="root of dataset.",
)
@click.option(
    "--partitioning-root",
    type=str,
    default="data",
    show_default=True,
    help="root of partitioning.",
)
@click.option(
    "--algorithm",
    "-a",
    type=tuple,
    cls=OptionEatAll,
    default="FedAvg",
    show_default=True,
    help="federated learning algorithm.",
)
@click.option(
    "--model",
    "-m",
    type=tuple,
    cls=OptionEatAll,
    default="mlp_mnist",
    show_default=True,
    help="model architecture.",
)
@click.option(
    "--epochs",
    "-e",
    type=int,
    default=5,
    show_default=True,
    help="number of local epochs.",
)
@click.option(
    "--loss-fn",
    type=str,
    default="cross_entropy",
    show_default=True,
    help="loss function to use (defined under fedsim.scores).",
)
@click.option(
    "--batch-size",
    type=int,
    default=32,
    show_default=True,
    help="local batch size.",
)
@click.option(
    "--test-batch-size",
    type=int,
    default=64,
    show_default=True,
    help="inference batch size.",
)
@click.option(
    "--optimizer",
    type=tuple,
    cls=OptionEatAll,
    default=("SGD", "lr:1.0"),
    show_default=True,
    help="server optimizer",
)
@click.option(
    "--local-optimizer",
    type=tuple,
    cls=OptionEatAll,
    default=("SGD", "lr:0.1", "weight_decay:0.001"),
    show_default=True,
    help="local optimizer",
)
@click.option(
    "--lr-scheduler",
    type=tuple,
    cls=OptionEatAll,
    default=("StepLR", "step_size:1", "gamma:1.0"),
    show_default=True,
    help="lr scheduler for server optimizer",
)
@click.option(
    "--local-lr-scheduler",
    type=tuple,
    cls=OptionEatAll,
    default=("StepLR", "step_size:1", "gamma:1.0"),
    show_default=True,
    help="lr scheduler for server optimizer",
)
@click.option(
    "--r2r-local-lr-scheduler",
    type=tuple,
    cls=OptionEatAll,
    default=("StepLR", "step_size:1", "gamma:0.999"),
    show_default=True,
    help="lr scheduler for round to round local optimization",
)
@click.option(
    "--pseed",
    "-p",
    type=int,
    default=0,
    show_default=True,
    help="seed for data partitioning.",
)
@click.option(
    "--seed",
    "-s",
    type=int,
    default=None,
    help="seed for random generators after data is partitioned.",
)
@click.option(
    "--device",
    type=str,
    default=None,
    show_default=True,
    help="device to load model and data one",
)
@click.option(
    "--log-dir",
    type=click.Path(resolve_path=True),
    default=None,
    show_default=True,
    help="directory to store the logs.",
)
@click.option(
    "--log-freq",
    type=int,
    default=50,
    show_default=True,
    help="gap between two reports in rounds.",
)
@click.option(
    "--train-report-point",
    type=int,
    default=10,
    show_default=True,
    help="number of last score reports points to store and get average performance.",
)
@click.option(
    "--verbosity",
    "-v",
    type=int,
    default=0,
    help="verbosity.",
    show_default=True,
)
@click.pass_context
def fed_learn(
    ctx: click.core.Context,
    rounds: int,
    data_manager: str,
    num_clients: int,
    client_sample_scheme: str,
    client_sample_rate: float,
    dataset_root: str,
    partitioning_root: str,
    algorithm: str,
    model: str,
    epochs: int,
    loss_fn: str,
    batch_size: int,
    test_batch_size: int,
    optimizer,
    local_optimizer,
    lr_scheduler,
    local_lr_scheduler,
    r2r_local_lr_scheduler,
    pseed: int,
    seed: Optional[float],
    device: Optional[str],
    log_dir: str,
    log_freq: int,
    train_report_point: int,
    verbosity: int,
) -> None:
    """simulates federated learning!

    .. note::

        To automatically include your custom data-manager by the provided cli tool,
        you can place your class in a python and pass its path to `-a` or
        `--data-manager` option (without .py) followed by column and name of the
        data-manager.
        For example, if you have data-manager `DataManager` stored in
        `foo/bar/my_custom_dm.py`, you can pass
        `--data-manager foo/bar/my_custom_dm:DataManager`.

    .. note::

        Arguments of the **init** method of any data-manager could be given in
        `arg:value` format following its name (or `path` if a local file is provided).
        Examples:

        .. code-block:: bash

            fedsim-cli fed-learn --data-manager BasicDataManager num_clients:1100 ...

        .. code-block:: bash

            fedsim-cli fed-learn --data-manager foo/bar/my_custom_dm:DataManager
            arg1:value ...


    .. note::

        To automatically include your custom algorithm by the provided cli tool,
        you can place your class in a python and pass its path to `-a` or
        `--algorithm` option (without .py) followed by column and name of the
        algorithm.
        For example, if you have algorithm `CustomFLAlgorithm` stored in
        `foo/bar/my_custom_alg.py`, you can pass
        `--algorithm foo/bar/my_custom_alg:CustomFLAlgorithm`.
        To automatically include your custom algorithm by the provided cli tool,
        you can place your class in a python and pass its path to `-a` or `--algorithm`
        option (without .py) followed by column and name of the algorithm.
        For example, if you have algorithm `CustomFLAlgorithm` stored in a
        `foo/bar/my_custom_alg.py`, you can pass
        `--algorithm foo/bar/my_custom_alg:CustomFLAlgorithm`.

    .. note::

        Arguments of the **init** method of any algoritthm could be given in
        `arg:value` format following its name (or `path` if a local file is provided).
        Examples:

        .. code-block:: bash

            fedsim-cli fed-learn --algorithm AdaBest mu:0.01 beta:0.6 ...

        .. code-block:: bash

            fedsim-cli fed-learn --algorithm foo/bar/my_custom_alg:CustomFLAlgorithm
            mu:0.01 ...

    .. note::

    To automatically include your custom model by the provided cli tool, you can place
    your class in a python and pass its path to `-m` or `--model` option (without .py)
    followed by column and name of the model.
    For example, if you have model `CustomModel` stored in a
    `foo/bar/my_custom_model.py`, you can pass
    `--model foo/bar/my_custom_alg:CustomModel`.

    .. note::

        Arguments of the **init** method of any model could be given in
        `arg:value` format following its name (or `path` if a local file is provided).
        Examples:

        .. code-block:: bash

            fedsim-cli fed-learn --model cnn_mnist num_classes:8 ...

        .. code-block:: bash

            fedsim-cli fed-learn --model foo/bar/my_custom_alg:CustomModel
            num_classes:8 ...


    """
    tb_logger = TensorboardLogger(path=log_dir)
    log_dir = tb_logger.get_dir()
    print("log available at %s", os.path.join(log_dir, "log.log"))
    print(
        "run the following for monitoring:\n\t tensorboard --logdir=%s",
        log_dir,
    )
    logging.basicConfig(
        filename=os.path.join(log_dir, "log.log"),
        level=verbosity * 10,
    )

    data_manager, data_manager_args = decode_margs(data_manager)
    data_manager_class = get_definition(
        name=data_manager,
        modules="fedsim.distributed.data_management",
    )

    algorithm, algorithm_args = decode_margs(algorithm)
    algorithm_class = get_definition(
        name=algorithm,
        modules=[
            "fedsim.distributed.centralized.training",
            "fedsim.distributed.decentralized.training",
        ],
    )

    model, model_args = decode_margs(model)
    model_class = get_definition(
        name=model,
        modules="fedsim.models",
    )
    model_class = partial(model_class, **model_args)

    optimizer, optimizer_args = decode_margs(optimizer)
    optimizer_class = get_definition(
        name=optimizer,
        modules="torch.optim",
    )
    optimizer_class = partial(optimizer_class, **optimizer_args)

    local_optimizer, local_optimizer_args = decode_margs(local_optimizer)
    local_optimizer_class = get_definition(
        name=local_optimizer,
        modules="torch.optim",
    )
    local_optimizer_class = partial(local_optimizer_class, **local_optimizer_args)

    lr_scheduler, lr_scheduler_args = decode_margs(lr_scheduler)
    lr_scheduler_class = get_definition(
        name=lr_scheduler,
        modules="torch.optim.lr_scheduler",
    )
    lr_scheduler_class = partial(lr_scheduler_class, **lr_scheduler_args)

    local_lr_scheduler, local_lr_scheduler_args = decode_margs(local_lr_scheduler)
    local_lr_scheduler_class = get_definition(
        name=local_lr_scheduler,
        modules="torch.optim.lr_scheduler",
    )
    local_lr_scheduler_class = partial(
        local_lr_scheduler_class,
        **local_lr_scheduler_args,
    )

    r2r_local_lr_scheduler, r2r_local_lr_scheduler_args = decode_margs(
        r2r_local_lr_scheduler
    )
    r2r_local_lr_scheduler_class = get_definition(
        name=r2r_local_lr_scheduler,
        modules="fedsim.lr_schedulers",
    )
    r2r_local_lr_scheduler_class = partial(
        r2r_local_lr_scheduler_class,
        **r2r_local_lr_scheduler_args,
    )

    data_manager_default_args = dict(
        root=dataset_root,
        num_clients=num_clients,
        seed=pseed,
        save_dir=partitioning_root,
    )
    data_manager_instant = data_manager_class(
        **{
            **data_manager_default_args,
            **data_manager_args,
        }
    )

    model_class = partial(model_class, **model_args)

    loss_criterion = None
    if hasattr(scores, loss_fn):
        loss_criterion = getattr(scores, loss_fn)
    else:
        raise Exception(f"loss_fn {loss_fn} is not defined in fedsim.scores")

    # set the device if it is not already set
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    cfg = {
        **ctx.params,
        **dict(
            data_manager=data_manager,
            algorithm=algorithm,
            model=model,
            optimizer=optimizer,
            local_optimizer=local_optimizer,
            lr_scheduler=lr_scheduler,
            local_lr_scheduler=local_lr_scheduler,
            r2r_local_lr_scheduler=r2r_local_lr_scheduler,
            device=device,
            log_dir=log_dir,

            data_manager_args = data_manager_args,
            algorithm_args=algorithm_args,
            model_args=model_args,
            optimizer_args=optimizer_args,
            local_optimizer_args=local_optimizer_args,
            lr_scheduler_args=lr_scheduler_args,
            local_lr_scheduler_args=local_lr_scheduler_args,
            r2r_local_lr_scheduler_args=r2r_local_lr_scheduler_args,

        )
    }
    logging.info("arguments: " + pformat(cfg))
    
    # set the seed of random generators
    if seed is not None:
        set_seed(seed, device)

    algorithm_instance = algorithm_class(
        data_manager=data_manager_instant,
        num_clients=num_clients,
        sample_scheme=client_sample_scheme,
        sample_rate=client_sample_rate,
        model_class=model_class,
        epochs=epochs,
        loss_fn=loss_criterion,
        optimizer_class=optimizer_class,
        local_optimizer_class=local_optimizer_class,
        lr_scheduler_class=lr_scheduler_class,
        local_lr_scheduler_class=local_lr_scheduler_class,
        r2r_local_lr_scheduler_class=r2r_local_lr_scheduler_class,
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        metric_logger=tb_logger,
        device=device,
        log_freq=log_freq,
        **algorithm_args,
    )
    algorithm_instance.hook_global_score_function("test", "accuracy", scores.accuracy)
    for key in data_manager_instant.get_local_splits_names():
        algorithm_instance.hook_local_score_function(key, "accuracy", scores.accuracy)

    logging.info(
        f"average of the last {train_report_point}\
            reports: {algorithm_instance.train(rounds, train_report_point)}"
    )
    tb_logger.flush()
