r"""
fed-learn cli Command
---------------------
"""

import logging
import os
from functools import partial
from pprint import pformat
from typing import Iterable
from typing import Optional

import click
import torch
from logall import TensorboardLogger

from fedsim import __version__ as fedsim_version
from fedsim.utils import set_seed

from .utils import OptionEatAll
from .utils import ingest_fed_context
from .utils import ingest_scores
from .utils import validate_score


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
    default=("BasicDataManager",),
    help="name of data manager.",
)
@click.option(
    "--train-split-name",
    type=str,
    default="train",
    show_default=True,
    help="name of local split to train train on",
)
@click.option(
    "--n-clients",
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
    "--algorithm",
    "-a",
    type=tuple,
    cls=OptionEatAll,
    default=("FedAvg",),
    show_default=True,
    help="federated learning algorithm.",
)
@click.option(
    "--model",
    "-m",
    type=tuple,
    cls=OptionEatAll,
    default=("mlp_mnist",),
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
    "--criterion",
    type=tuple,
    cls=OptionEatAll,
    default=("CrossEntropyLoss", "log_freq:50"),
    show_default=True,
    help="loss function to use (defined under fedsim.losses).",
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
    default=("StepLR", "step_size:1", "gamma:1"),
    show_default=True,
    help="lr scheduler for round to round local optimization",
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
    "--n-point-summary",
    type=int,
    default=10,
    show_default=True,
    help="number of last score report points to store and get the final average\
        performance from.",
)
@click.option(
    "--local-score",
    type=tuple,
    cls=OptionEatAll,
    multiple=True,
    default=(("Accuracy", "log_freq:50", "split:train"),),
    help="hooks a score object to a split of local datasets. Choose the score classes\
        from `fedsim.scores`. It is possible to call this option multiple times.",
)
@click.option(
    "--global-score",
    type=tuple,
    cls=OptionEatAll,
    multiple=True,
    default=(("CrossEntropyScore", "log_freq:50", "split:test"),),
    help="hooks a score object to a split of global datasets. Choose the score classes\
        from `fedsim.scores`. It is possible to call this option multiple times.",
)
@click.pass_context
def fed_learn(
    ctx: click.core.Context,
    rounds: int,
    data_manager: str,
    train_split_name: str,
    n_clients: int,
    client_sample_scheme: str,
    client_sample_rate: float,
    algorithm: str,
    model: str,
    epochs: int,
    criterion: str,
    batch_size: int,
    test_batch_size: int,
    optimizer,
    local_optimizer,
    lr_scheduler,
    local_lr_scheduler,
    r2r_local_lr_scheduler,
    seed: Optional[float],
    device: Optional[str],
    log_dir: str,
    n_point_summary: int,
    local_score: Iterable,
    global_score: Iterable,
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
    print(f'log available at: \n\t {os.path.join(log_dir, "log.log")}')
    print(f"run the following for monitoring:\n\t tensorboard --logdir={log_dir}")
    print()
    log_handler = logging.FileHandler(os.path.join(log_dir, "log.log"))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(log_handler)

    # set the device if it is not already set
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    cfg = ingest_fed_context(
        data_manager,
        algorithm,
        model,
        criterion,
        optimizer,
        local_optimizer,
        lr_scheduler,
        local_lr_scheduler,
        r2r_local_lr_scheduler,
    )

    # log configuration
    args_dict = {
        def_name: {defn.name: defn.arguments} for def_name, defn in cfg.items()
    }
    log = {**ctx.params, **args_dict}
    log["device"] = device
    log["log_dir"] = log_dir
    log["fedsim_version"] = fedsim_version
    logger.info("arguments: \n" + pformat(log))
    tb_logger.get_logger_object().add_text("config", f"{log}")

    # set the seed of random generators
    if seed is not None:
        set_seed(seed, device)

    # set the train split as the split of criterion (loss)
    criterion_def = partial(cfg["criterion"].definition, split=train_split_name)

    data_manager_instant = cfg["data_manager"].definition()

    algorithm_instance = cfg["algorithm"].definition(
        data_manager=data_manager_instant,
        metric_logger=tb_logger,
        num_clients=n_clients,
        sample_scheme=client_sample_scheme,
        sample_rate=client_sample_rate,
        model_def=cfg["model"].definition,
        epochs=epochs,
        criterion_def=criterion_def,
        optimizer_def=cfg["optimizer"].definition,
        local_optimizer_def=cfg["local_optimizer"].definition,
        lr_scheduler_def=cfg["lr_scheduler"].definition,
        local_lr_scheduler_def=cfg["local_lr_scheduler"].definition,
        r2r_local_lr_scheduler_def=cfg["r2r_local_lr_scheduler"].definition,
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        device=device,
    )

    local_score_defs = ingest_scores(local_score)
    global_score_defs = ingest_scores(global_score)

    local_score_defs = [score_obj.definition for score_obj in local_score_defs]
    global_score_defs = [score_obj.definition for score_obj in global_score_defs]

    for l_score in local_score_defs:
        local_split_names = data_manager_instant.get_local_splits_names()
        split_name, score_name = validate_score(
            l_score, local_split_names, mode="local"
        )
        algorithm_instance.hook_local_score(
            l_score,
            split_name=split_name,
            score_name=score_name,
        )

    for g_score in global_score_defs:
        global_split_names = data_manager_instant.get_global_splits_names()
        split_name, score_name = validate_score(
            g_score, global_split_names, mode="global"
        )
        algorithm_instance.hook_global_score(
            g_score,
            split_name=split_name,
            score_name=score_name,
        )

    report_summary = algorithm_instance.train(rounds, n_point_summary, train_split_name)
    logger.info(f"average of the last {n_point_summary} reports")
    logger.info(report_summary)
    tb_logger.flush()


if __name__ == "__main__":
    fed_learn()
