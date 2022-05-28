from email.policy import default
from logging import root
from typing import AnyStr, Optional
import os
import click
from torch.utils.tensorboard import SummaryWriter
#
from data_manager.feddyn_data_manager import FedDynDataManager
from utils import get_from_module, search_in_submodules, set_seed
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
    '--data-manager', type=str, 
    default='FedDynDataManager', help='name of data manager.',
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
    '--dataset', '-dst', type=str, default='mnist', help='name of dataset.',
    )
@click.option(
    '--dataset-root', '-drt', type=str, 
    default='data', help='root of dataset.',
    )
@click.option(
    '--partitioning-root', '-prt', type=str, help='root of partitioning.',
    default='data'
    )
@click.option(
    '--partitioning-rule', '-prl', type=click.Choice(['dir', 'iid']), 
    default='iid', help='partitioning rule.',
    )
@click.option(
    '--sample-balance', '-b', type=float, default=0, show_default=True,
    help='balance of the number of samples per client \
        (0 is balanced, 1 is very unbalanced).',
    )
@click.option(
    '--label-balance', '-lb', type=float, default=0.03, show_default=True,
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
    '--model', '-m', 
    type=str, 
    default='mlp_mnist', 
    show_default=True,
    help='model architecture.',
    )
@click.option(
    '--epochs', '-e', 
    type=int, 
    default=5, 
    show_default=True,
    help='number of local epochs.',
    )
@click.option(
    '--loss-fn', '-l', 
    type=click.Choice(['ce', 'mse']), 
    default='ce', 
    show_default=True,
    help='loss function to use (se stands for cross-entropy).',
    )
@click.option(
    '--batch-size', '-bs', 
    type=int, 
    default=32, 
    show_default=True,
    help='local batch size.',
    )
@click.option(
    '--test-batch-size', '-tbs', 
    type=int, 
    default=64, 
    show_default=True,
    help='inference batch size.',
    )
@click.option(
    '--local_weight_decay', '-wd', 
    type=float, 
    default=0.001, 
    show_default=True,
    help='local weight decay.',
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
    '--clr-decay', '-cd', 
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
    '--min-clr', '-mc', 
    type=float, 
    default=1e-12, 
    show_default=True,
    help='minimum client leanring rate.',
    )
@click.option(
    '--clr-step-size', '-cst', 
    type=int, 
    default=1, 
    help='step size for clr decay (in rounds), used both with cosine and step',
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
    '--device', '-d', 
    type=click.Choice(['cpu','cuda','0', '1', '2', '3', '4', '5', '6', '7']), 
    default='cuda', 
    help='device to load model and data one',
    )
@click.option(
    '--log-dir', '-ldir', 
    type=click.Path(resolve_path=True), 
    default=None, 
    help='directory to store the tensorboard logs.',
    )
@click.option(
    '--verbosity', '-v', 
    type=int, 
    default=0, 
    help='verbosity.',
    )
@click.pass_context
def fed_learn(
    ctx, rounds: int, data_manager: str, num_clients: int, 
    client_sample_scheme: str, client_sample_rate: float, dataset: str, 
    dataset_root: str, partitioning_root: str, partitioning_rule: str, 
    sample_balance: float, label_balance: float, algorithm: str, model: str, 
    epochs: int, loss_fn: str, batch_size: int, test_batch_size: int,
    local_weight_decay: float, clr: float, slr: float, clr_decay: float,
    clr_decay_type: str, min_clr: float, clr_step_size: int, pseed: int, 
    seed: float, device: str, log_dir: str, verbosity: int
    ) -> object:
    """_summary_

    Args:
        ctx (_type_): for extra parameters passed to click
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
    data_manager_class = search_in_submodules('data_manager', data_manager)
    # make data manager
    data_manager_instant = data_manager_class(
        dataset_root, dataset, num_clients, partitioning_rule, sample_balance, 
        label_balance, pseed, partitioning_root,
        )
    # set the seed of random generators
    set_seed(seed, device)

    algorithm_class = get_from_module(
        'federation.algorithms', algorithm, 'Algorithm'
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
        logger=SummaryWriter(log_dir),
        device=device,
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