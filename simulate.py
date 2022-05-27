from typing import AnyStr, Optional
import os
import click
import importlib
from torch.utils.tensorboard import SummaryWriter
#
from data_manager.feddyn_data_manager import FedDynDataManager

from federation import algorithms

# Enable click
os.environ['LC_ALL'] = 'C.UTF-8'
os.environ['LANG'] = 'C.UTF-8'

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
    '--rounds', '-r', type=int, default=100, show_default=True, 
    help='number of communication rounds.',
    )
@click.option(
    '--num-clients', '-n', type=int, default=500, help='number of clients.',
    )
@click.option(
    '--client-sample-scheme', '-css', type=str, default='uniform', 
    help='client sampling scheme.',
    )
@click.option(
    '--client-sample-rate', '-csr', type=float, default=0.01, 
    help='mean portion of num clients to sample.',
    )
@click.option(
    '--dataset', '-dst', type=str, default='cifar10', help='name of dataset.',
    )
@click.option(
    '--sample-balance', '-b', type=float, default=0, show_default=True,
    help='balance of the number of samples per client \
        (0 is balanced, 1 is very unbalanced).',
    )
@click.option(
    '--label-balance', '-l', type=float, default=0.03, show_default=True,
    help='balance of the labels from among clients\
         (closer to 0 is more heterogeneous).',
    )
@click.option(
    '--algorithm', '-a', 
    type=click.Choice(['fedavg', 'scaffold', 'feddyn', 'adabest']), 
    default='fedavg', 
    show_default=True,
    help='federated learning algorithm.',
    )
@click.option(
    '--clr', '-c', 
    type=float, 
    default=0.05, 
    show_default=True,
    help='client learning rate.',
    )
@click.option(
    '--slr', '-slr', 
    type=float, 
    default=1.0, 
    show_default=True,
    help='server learning rarte.',
    )
@click.option(
    '--clr-decay', '-d', 
    type=float, 
    default=1.0,
    show_default=True,
    help='scalar for round to round decay of the client learning rate.',
    )
@click.option(
    '--clr-decay-type', '-t',
    type=click.Choice(['step', 'cosine']), 
    default='step', 
    show_default=True,
    help='type of decay for client learning rate decay.',
    )
@click.option(
    '--min-clr', '-m', 
    type=float, 
    default=1e-12, 
    show_default=True,
    help='minimum client leanring rate.',
    )
@click.option(
    '--pseed', '-p', 
    type=int, 
    default=0, 
    help='seed for data partitioning.',
    )
@click.option(
    '--seed', '-s', 
    type=int, 
    default=0, 
    help='seed for random generators after data is partitioned.',
    )
@click.option(
    '--log-dir', '-ldir', 
    type=click.Path(resolve_path=True), 
    default=None, 
    help='directory to store the tensorboard logs.',
    )
@click.option(
    '--verbosity', '-v', 
    type=click.Choice([0, 1, 2]), 
    default=0, 
    help='verbosity.',
    )
def fed_learn(
    ctx, rounds: int, num_clients: int, client_sample_scheme: str, 
    client_sample_rate: float, dataset: str, sample_balance: float, 
    label_balance: float, algorithm: AnyStr, clr: float, slr: float,
    clr_decay: float, clr_decay_type: AnyStr, min_clr: float, pseed: int, 
    seed: float, log_dir: str, verbosity: int) -> object:
    """_summary_

    Args:
        ctx (_type_): for extra parameters passed to click
        rounds (int): number of training rounds
        num_clients (int): number of clients
        client_sample_scheme (str): client sampling scheme (default: uniform)
        client_sample_rate (float): client sampling rate
        dataset (str): name of the dataset to train on
        sample_balance (float): balance of num samples on clients
        label_balance (float): balance of labels among clients
        algorithm (AnyStr): federated learning algorithm to use for training
        clr (float): client learning rate
        slr (float): server learning rate
        clr_decay (float): decay of the client learning rate through rounds
        clr_decay_type (AnyStr): type of clr decay (step or cosine)
        min_clr (float): min clr
        pseed (int): partitioning random seed
        seed (float): seed of the training itself
        verbosity (int): verbosity of the outputs

    Raises:
        NotImplementedError: _description_

    Returns:
        object: average training loss of the last rounds
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

    # make data manager
    data_manager = FedDynDataManager(
        dataset, num_clients, sample_balance, label_balance, pseed
        )
    # set the seed of random generators
    # TODO: set the seed of random generators

    # setup algorithm
    if algorithm in algorithms.classes:
        algorithm_module = importlib.import_module(
            'federation.algorithms.{}'.format(algorithm)
            )
    else:
        raise NotImplementedError

    algorithm_class = getattr(algorithm_module, 'Algorithm')

    algorithm_instance = algorithm_class(
        data_manager=data_manager, 
        num_clients=num_clients, 
        sample_scheme=client_sample_scheme, 
        sample_rate=client_sample_rate, 
        slr=slr, 
        clr=clr, 
        clr_decay=clr_decay,
        clr_decay_type=clr_decay_type, 
        min_clr=min_clr,
        algorithm_params=algorithm_params,
        logger=SummaryWriter(log_dir),
        verbosity=verbosity,
        )

    loss = algorithm_instance.train(rounds)
    print(loss)





def main():
    """ main fn

    """
    run()


if __name__ == '__main__':
    main()