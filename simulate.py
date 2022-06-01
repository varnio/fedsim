from typing import TypeVar, Optional
import os
import click
from torch.utils.tensorboard import SummaryWriter
#
from utils import get_from_module, search_in_submodules, set_seed

# Enable click
os.environ['LC_ALL'] = 'C.UTF-8'
os.environ['LANG'] = 'C.UTF-8'

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

@click.group()
def run():
    pass

@run.command(
    name='fed-learn', 
    context_settings = dict(
        ignore_unknown_options=True, 
        allow_extra_args=True,
    )
)
@click.option(
    '--rounds', 
    '-r', 
    type=int, 
    default=100, 
    show_default=True, 
    help='number of communication rounds.',
)
@click.option(
    '--data-manager', 
    type=str, 
    show_default=True,
    default='FedDynDataManager', 
    help='name of data manager.',
)
@click.option(
    '--num-clients', 
    '-n', 
    type=int, 
    default=500, 
    show_default=True,
    help='number of clients.',
)
@click.option(
    '--client-sample-scheme', 
    type=str, 
    default='uniform', 
    show_default=True,
    help='client sampling scheme.',
)
@click.option(
    '--client-sample-rate', '-c', 
    type=float, 
    default=0.01,
    show_default=True,
    help='mean portion of num clients to sample.',
)
@click.option(
    '--dataset', '-d', 
    type=str, 
    default='mnist', 
    show_default=True,
    help='name of dataset.',
)
@click.option(
    '--dataset-root', 
    type=str, 
    default='data', 
    show_default=True,
    help='root of dataset.',
)
@click.option(
    '--partitioning-root', 
    type=str, 
    default='data', 
    show_default=True,
    help='root of partitioning.',
)
@click.option(
    '--partitioning-rule', 
    type=str, 
    default='iid', 
    show_default=True,
    help='partitioning rule.',
)
@click.option(
    '--sample-balance', 
    type=float, 
    default=0, 
    show_default=True,
    help='balance of the number of samples per client \
        (0 is balanced, 1 is very unbalanced).',
)
@click.option(
    '--label-balance', 
    type=float, 
    default=0.03, 
    show_default=True,
    help='balance of the labels from among clients \
        (closer to 0 is more heterogeneous).',
)
@click.option(
    '--algorithm', 
    '-a', 
    type=str, 
    default='fedavg', 
    show_default=True,
    help='federated learning algorithm.',
)
@click.option(
    '--model', 
    '-m', 
    type=str, 
    default='mlp_mnist', 
    show_default=True,
    help='model architecture.',
)
@click.option(
    '--epochs', 
    '-e', 
    type=int, 
    default=5, 
    show_default=True,
    help='number of local epochs.',
)
@click.option(
    '--loss-fn', 
    type=click.Choice(['ce', 'mse']), 
    default='ce', 
    show_default=True,
    help='loss function to use (se stands for cross-entropy).',
)
@click.option(
    '--batch-size', 
    type=int, 
    default=32, 
    show_default=True,
    help='local batch size.',
)
@click.option(
    '--test-batch-size', 
    type=int, 
    default=64, 
    show_default=True,
    help='inference batch size.',
)
@click.option(
    '--local_weight_decay', 
    type=float, 
    default=0.001, 
    show_default=True,
    help='local weight decay.',
)
@click.option(
    '--clr', 
    '-l', 
    type=float, 
    default=0.05, 
    show_default=True,
    help='client learning rate.',
)
@click.option(
    '--slr', 
    type=float, 
    default=1.0, 
    show_default=True,
    help='server learning rarte.',
)
@click.option(
    '--clr-decay', 
    type=float, 
    default=1.0, 
    show_default=True,
    help='scalar for round to round decay of the client learning rate.',
)
@click.option(
    '--clr-decay-type', 
    type=click.Choice(['step', 'cosine']), 
    default='step', 
    show_default=True,
    help='type of decay for client learning rate decay.',
)
@click.option(
    '--min-clr', 
    type=float, 
    default=1e-12, 
    show_default=True,
    help='minimum client leanring rate.',
)
@click.option(
    '--clr-step-size',
    type=int, 
    default=1, 
    show_default=True,
    help='step size for clr decay (in rounds), used both with cosine and step',
)
@click.option(
    '--pseed', 
    '-p', 
    type=int, 
    default=0, 
    show_default=True,
    help='seed for data partitioning.',
)
@click.option(
    '--seed', 
    '-s', 
    type=int, 
    default=None, 
    help='seed for random generators after data is partitioned.',
)
@click.option(
    '--device', 
    type=str, 
    default='cuda', 
    show_default=True,
    help='device to load model and data one',
)
@click.option(
    '--log-dir', 
    type=click.Path(resolve_path=True), 
    default=None, 
    show_default=True,
    help='directory to store the tensorboard logs.',
)
@click.option(
    '--log-freq', 
    type=int, 
    default=50, 
    show_default=True,
    help='gap between two reports in rounds.',
)
@click.option(
    '--verbosity', 
    '-v', 
    type=int, 
    default=0, 
    help='verbosity.',
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
    dataset: str, 
    dataset_root: str, 
    partitioning_root: str, 
    partitioning_rule: str, 
    sample_balance: float, 
    label_balance: float, 
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
    verbosity: int
) -> None:
    """simulates federated learning!

    Args:
        ctx (click.core.Context): for extra parameters passed to click
        rounds (int): number of training rounds
        data_manager (str): name of data manager found under data_manager dir
        num_clients (int): number of clients
        client_sample_scheme (str): client sampling scheme (default: uniform)
        client_sample_rate (float): client sampling rate
        dataset (str): name of the dataset to train on
        dataset_root (str): root of the dataset
        partitioning_root (str): root of partitioning destination
        partitioning_rule (str): rule of partitioning ('dir', 'iid')
        sample_balance (float): balance of num samples on clients
        label_balance (float): balance of labels among clients
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
    # get algorithm specific parameters
    algorithm_params = dict()
    i = 0
    while i < len(ctx.args):
        if i == len(ctx.args) - 1 or ctx.args[i+1][:2] == '--':
            algorithm_params[ctx.args[i][2:]] = 'True'
            i += 1
        else:
            algorithm_params[ctx.args[i][2:]] = ctx.args[i + 1]
            i +=2
    data_manager_class = search_in_submodules('data_manager', data_manager)
    # make data manager
    data_manager_instant = data_manager_class(
        dataset_root, dataset, num_clients, partitioning_rule, sample_balance, 
        label_balance, pseed, partitioning_root,
        )
    # set the seed of random generators
    if seed is  not None:
        set_seed(seed, device)

    algorithm_class = get_from_module(
        'federation.algorithms', 
        algorithm, 
        'Algorithm',
    )
    algorithm_instance = algorithm_class(
        data_manager=data_manager_instant, 
        num_clients=num_clients, 
        sample_scheme=client_sample_scheme, 
        sample_rate=client_sample_rate, 
        model=model,
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
        algorithm_params=algorithm_params,
        metric_logger=SummaryWriter(log_dir),
        device=device,
        log_freq=log_freq,
        verbosity=verbosity,
    )

    alg_ret = algorithm_instance.train(rounds)
    # click.echo(ret)


def main():
    """ main fn

    """
    run()


if __name__ == '__main__':
    main()
