import click
import os
from typing import Optional
from torch.utils.tensorboard import SummaryWriter
from fedsim.utils import search_in_submodules, set_seed
import inspect
import yaml
from collections import namedtuple
import logging
from pprint import pformat


@click.command(
    name="fed-learn",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ),
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
    type=str,
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
    type=str,
    default="FedAvg",
    show_default=True,
    help="federated learning algorithm.",
)
@click.option(
    "--model",
    "-m",
    type=str,
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
    type=click.Choice(["ce", "mse"]),
    default="ce",
    show_default=True,
    help="loss function to use (se stands for cross-entropy).",
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
    "--local-weight-decay",
    type=float,
    default=0.001,
    show_default=True,
    help="local weight decay.",
)
@click.option(
    "--clr",
    "-l",
    type=float,
    default=0.05,
    show_default=True,
    help="client learning rate.",
)
@click.option(
    "--slr",
    type=float,
    default=1.0,
    show_default=True,
    help="server learning rarte.",
)
@click.option(
    "--clr-decay",
    type=float,
    default=1.0,
    show_default=True,
    help="scalar for round to round decay of the client learning rate.",
)
@click.option(
    "--clr-decay-type",
    type=click.Choice(["step", "cosine"]),
    default="step",
    show_default=True,
    help="type of decay for client learning rate decay.",
)
@click.option(
    "--min-clr",
    type=float,
    default=1e-12,
    show_default=True,
    help="minimum client leanring rate.",
)
@click.option(
    "--clr-step-size",
    type=int,
    default=1,
    show_default=True,
    help="step size for clr decay (in rounds), used both with cosine and step",
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
    default="cuda",
    show_default=True,
    help="device to load model and data one",
)
@click.option(
    "--log-dir",
    type=click.Path(resolve_path=True),
    default=None,
    show_default=True,
    help="directory to store the tensorboard logs.",
)
@click.option(
    "--log-freq",
    type=int,
    default=50,
    show_default=True,
    help="gap between two reports in rounds.",
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
    local_weight_decay: float,
    clr: float,
    slr: float,
    clr_decay: float,
    clr_decay_type: str,
    min_clr: float,
    clr_step_size: int,
    pseed: int,
    seed: Optional[float],
    device: str,
    log_dir: str,
    log_freq: int,
    verbosity: int,
) -> None:
    """simulates federated learning!
        - Additional arg for Algorithm is specified using prefix `--a-`
        - Additional arg for DataManager is specified using prefix `--d-`

    Args:
        ctx (click.core.Context): for extra parameters passed to click
        rounds (int): number of training rounds
        data_manager (str): name of data manager found under data_manager dir
        num_clients (int): number of clients
        client_sample_scheme (str): client sampling scheme (default: uniform)
        client_sample_rate (float): client sampling rate
        dataset_root (str): root of the dataset
        partitioning_root (str): root of partitioning destination
        algorithm (str): federated learning algorithm to use for training
        model (str): model architecture
        epochs (int): number of local epochs
        batch_size (int): batch size for local optimization
        test_batch_size (int): batch size for inference
        local_weight_decay (float): weight decay for local optimization
        clr (float): client learning rate
        slr (float): server learning rate
        clr_decay (float): decay of the client learning rate through rounds
        clr_decay_type (str): type of clr decay (step or cosine)
        min_clr (float): min clr
        clr_step_size (int): step size for clr decay (in rounds)
        pseed (int): partitioning random seed
        seed (float): seed of the training itself
        device (str): device to load model and data on
        log_dir (str): the directory to store logs
        log_freq (int): gap between two reports in rounds.
        verbosity (int): verbosity of the outputs
    """
    summary_writer = SummaryWriter(log_dir)
    log_dir = summary_writer.get_logdir()
    print("log available at %s", os.path.join(log_dir, "log.log"))
    print(
        "run the following for monitoring:\n\t tensorboard --logdir=%s",
        log_dir,
    )
    logging.basicConfig(
        filename=os.path.join(log_dir, "log.log"),
        level=verbosity * 10,
    )
    logging.info("arguments: " + pformat(ctx.params))
    # get the classes
    data_manager_class = search_in_submodules(
        "fedsim.data_manager", data_manager
    )
    if data_manager_class is None:
        pass
    algorithm_class = search_in_submodules("fedsim.fl.algorithms", algorithm)
    if algorithm_class is None:
        algorithm_class = search_in_submodules(
            "fedsim.fl.mtl_algorithms", algorithm
        )

    dtm_args = dict()
    alg_args = dict()

    ClassContext = namedtuple("ClassContext", ["cls", "prefix", "arg_dict"])

    context_pool = dict(
        alg_context=ClassContext(algorithm_class, "a-", alg_args),
        dtm_context=ClassContext(data_manager_class, "d-", dtm_args),
    )

    def add_arg(key, value, prefix):
        context = list(
            filter(lambda x: x.prefix == prefix, context_pool.values())
        )
        if len(context) == 0:
            raise Exception("{} is an invalid argument".format(key))
        else:
            context = context[0]
        if key in inspect.signature(context.cls).parameters.keys():
            context.arg_dict[key] = yaml.safe_load(value)
        else:
            raise Exception(
                "{} is not an argument of {}".format(key, context.cls.__name__)
            )

    i = 0
    while i < len(ctx.args):
        if ctx.args[i][:2] != "--":
            raise Exception("unexpected option {}".format(ctx.args[i]))
        if ctx.args[i][2] == "-":
            raise Exception(
                "option {} is not valid. No option should starts with ---".format(
                    ctx.args[i]
                )
            )
        prefix = ctx.args[i][2:4]
        arg = ctx.args[i][4:]
        arg = arg.replace("-", "_")
        if i == len(ctx.args) - 1 or ctx.args[i + 1][:2] == "--":
            add_arg(arg, "True", prefix)
            i += 1
        else:
            next_arg = ctx.args[i + 1]
            add_arg(arg, next_arg, prefix)
            i += 2

    data_manager_args = dict(
        root=dataset_root,
        num_clients=num_clients,
        seed=pseed,
        save_path=partitioning_root,
    )
    data_manager_instant = data_manager_class(
        **{
            **data_manager_args,
            **context_pool["dtm_context"].arg_dict,
        }
    )

    # set the seed of random generators
    if seed is not None:
        set_seed(seed, device)

    algorithm_instance = algorithm_class(
        data_manager=data_manager_instant,
        num_clients=num_clients,
        sample_scheme=client_sample_scheme,
        sample_rate=client_sample_rate,
        model_class=model,
        epochs=epochs,
        loss_fn=loss_fn,
        batch_size=batch_size,
        test_batch_size=test_batch_size,
        local_weight_decay=local_weight_decay,
        slr=slr,
        clr=clr,
        clr_decay=clr_decay,
        clr_decay_type=clr_decay_type,
        min_clr=min_clr,
        clr_step_size=clr_step_size,
        metric_logger=summary_writer,
        device=device,
        log_freq=log_freq,
        **context_pool["alg_context"].arg_dict,
    )

    alg_ret = algorithm_instance.train(rounds)
    summary_writer.flush()
    # click.echo(ret)
