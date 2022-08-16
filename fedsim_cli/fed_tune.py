r"""
fed-tune cli Command
--------------------
"""

import logging
import os
from collections import OrderedDict
from functools import partial
from pprint import pformat
from typing import Iterable
from typing import Optional

import click
import torch
from logall import TensorboardLogger
from skopt import Optimizer
from skopt.space import Categorical
from skopt.space import Integer
from skopt.space import Real

from fedsim import __version__ as fedsim_version
from fedsim.utils import set_seed

from .utils import LogFilter
from .utils import OptionEatAll
from .utils import ingest_fed_context
from .utils import ingest_scores
from .utils import validate_score


@click.command(
    name="fed-tune",
    help="Tunes a Federated Learning system.",
)
@click.option(
    "--n-iters",
    type=int,
    default=10,
    show_default=True,
    help="number of iterations to ask and tell the skopt optimizer",
)
@click.option(
    "--skopt-n-initial-points",
    type=int,
    default=10,
    show_default=True,
    help="number of initial points for skopt optimizer",
)
@click.option(
    "--skopt-random-state",
    type=int,
    default=10,
    show_default=True,
    help="random state for skopt optimizer",
)
@click.option(
    "--skopt-base-estimator",
    type=click.Choice(["GP", "RF", "ET", "GBRT"]),
    default="GP",
    show_default=True,
    help="skopt estimator",
)
@click.option(
    "--eval-metric",
    type=str,
    default="server.avg.test.cross_entropy_score",
    show_default=True,
    help="complete name of the metric (returned from train method of algorithm) to\
        minimize (or maximize if --maximize is passed)",
)
@click.option(
    "--maximize/--minimize",
    default=False,
    show_default=False,
    help="complete name of the metric (returned from train method of algorithm) to\
        minimize or maximize",
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
    default=("StepLR", "step_size:1", "gamma:0.999"),
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
def fed_tune(
    ctx: click.core.Context,
    n_iters: int,
    skopt_n_initial_points: int,
    skopt_random_state: int,
    skopt_base_estimator: str,
    eval_metric: str,
    maximize: bool,
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
    log_handler.addFilter(LogFilter("parent"))
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

    local_score_defs = ingest_scores(local_score)
    global_score_defs = ingest_scores(global_score)

    local_score_defs = [score_obj.definition for score_obj in local_score_defs]
    global_score_defs = [score_obj.definition for score_obj in global_score_defs]

    hparams = OrderedDict()
    for def_name, defn in cfg.items():
        for arg_name, arg in defn.harguments.items():
            hparams[".".join([def_name, arg_name])] = arg
    if len(hparams) == 0:
        raise Exception("no hyper-params specified!")

    # log configuration
    compined_args = dict()
    for def_name, defn in cfg.items():
        compined_args[def_name] = {defn.name: {**defn.arguments, **defn.harguments}}
    log = {**ctx.params, **compined_args}
    log["device"] = device
    log["log_dir"] = log_dir
    log["fedsim_version"] = fedsim_version
    logger.info("configuration: \n" + pformat(log), extra={"flow": "parent"})
    logger.info("hyper-params: \n" + pformat(dict(hparams)), extra={"flow": "parent"})
    tb_logger.get_logger_object().add_text(" grid config", f"{log}")
    # make hparam opt
    optimizer = Optimizer(
        dimensions=hparams.values(),
        base_estimator=skopt_base_estimator,
        n_initial_points=skopt_n_initial_points,
        random_state=skopt_random_state,
    )

    def refine_hparams(hparam_dict, obj_name):
        """filters hparams of obj_name from hparam_dict and returns as a dict.

        Example:
            if hparam_dict := {'algorithm.mu': 0.1, 'model.omega: 5'},
            obj_name == algorithm, then the result would be {'mu': 0.1}

        """
        hargs = dict()
        for k, v in hparam_dict.items():
            o_name, arg_name = k.split(".")
            if o_name == obj_name:
                hargs[arg_name] = v
        return hargs

    def update_def(hparam_dict, obj_name):
        """updates the partial definition of cfg[obj_name].definition, with
        corresponding entries in hparam_dict.
        """
        return partial(
            cfg[obj_name].definition, **refine_hparams(hparam_dict, obj_name)
        )

    best_metric = None
    best_config = None
    best_itr = None
    # loop
    for itr in range(n_iters):
        suggested = optimizer.ask()
        hparams_suggested = {k: v for k, v in zip(hparams.keys(), suggested)}
        data_manager_def = update_def(hparams_suggested, "data_manager")
        algorithm_def = update_def(hparams_suggested, "algorithm")
        model_def = update_def(hparams_suggested, "model")
        criterion_def = update_def(hparams_suggested, "criterion")
        optimizer_def = update_def(hparams_suggested, "optimizer")
        local_optimizer_def = update_def(hparams_suggested, "local_optimizer")
        lr_scheduler_def = update_def(hparams_suggested, "lr_scheduler")
        local_lr_scheduler_def = update_def(hparams_suggested, "local_lr_scheduler")
        r2r_local_lr_scheduler_def = update_def(
            hparams_suggested, "r2r_local_lr_scheduler"
        )
        # logging
        identity = ""
        for k, v in hparams_suggested.items():
            if len(identity) > 0:
                identity += "&"
            if isinstance(hparams[k], Real):
                identity += f"{k}__{v:.3f}"
            elif isinstance(hparams[k], (Integer, Categorical)):
                identity += f"{k}__{v}"
            else:
                raise Exception(f"{k} is not a hyperparam!")
        child_dir = os.path.join(log_dir, str(itr), identity)
        tb_logger_child = TensorboardLogger(path=child_dir)
        child_dir = tb_logger_child.get_dir()
        print("log of child available at %s", os.path.join(child_dir, "log.log"))
        log_handler = logging.FileHandler(os.path.join(child_dir, "log.log"))
        log_handler.addFilter(LogFilter(identity))
        logger.addHandler(log_handler)

        compined_args = dict()
        for def_name, defn in cfg.items():
            compined_args[def_name] = {
                defn.name: {
                    **defn.arguments,
                    **refine_hparams(hparams_suggested, def_name),
                }
            }
        log = {**log, **compined_args}
        log["log_dir"] = log_dir
        log["fedsim_version"] = fedsim_version
        logger.info("configuration: \n" + pformat(log), extra={"flow": identity})
        tb_logger_child.get_logger_object().add_text("instance config", f"{log}")
        # set the seed of random generators
        if seed is not None:
            set_seed(seed, device)

        data_manager_instant = data_manager_def()

        algorithm_instance = algorithm_def(
            data_manager=data_manager_instant,
            metric_logger=tb_logger_child,
            num_clients=n_clients,
            sample_scheme=client_sample_scheme,
            sample_rate=client_sample_rate,
            model_def=model_def,
            epochs=epochs,
            criterion_def=criterion_def,
            optimizer_def=optimizer_def,
            local_optimizer_def=local_optimizer_def,
            lr_scheduler_def=lr_scheduler_def,
            local_lr_scheduler_def=local_lr_scheduler_def,
            r2r_local_lr_scheduler_def=r2r_local_lr_scheduler_def,
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            device=device,
        )

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

        report_summary = algorithm_instance.train(
            rounds,
            n_point_summary,
            train_split_name,
        )
        logger.info(
            f"average of the last {n_point_summary} reports", extra={"flow": identity}
        )
        logger.info(report_summary, extra={"flow": identity})
        tb_logger_child.flush()
        metric = report_summary[eval_metric]
        tb_logger.log_scalars(
            {f"grid.{k}": v for k, v in report_summary.items()}, step=itr
        )
        if maximize:
            metric = -metric
        optimizer.tell(suggested, metric)

        # save the best metric
        if best_metric is None or metric < best_metric:
            best_metric = metric
            best_config = identity
            best_itr = itr

    if maximize:
        best_metric = -best_metric
    logger.info(
        f"best metric observed ({best_metric:.3f}) at iteration {best_itr}",
        extra={"flow": "parent"},
    )
    best_conf_fmt = best_config.replace("__", "=").replace("&", ",")
    logger.info(
        f"best config {best_conf_fmt}",
        extra={"flow": "parent"},
    )
    tb_logger.get_logger_object().add_text("best config", f"{best_conf_fmt}")
