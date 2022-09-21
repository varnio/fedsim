r"""
FedNova
-------
"""
from .fedavg import FedAvg
from .utils import serial_aggregation


class FedNova(FedAvg):
    r"""Implements FedNova algorithm for centralized FL.

    For further details regarding the algorithm we refer to `Tackling the Objective
    Inconsistency Problem in Heterogeneous Federated Optimization`_.

    Args:
        data_manager (``distributed.data_management.DataManager``): data manager
        metric_logger (``logall.Logger``): metric logger for tracking.
        num_clients (int): number of clients
        sample_scheme (``str``): mode of sampling clients. Options are ``'uniform'``
            and ``'sequential'``
        sample_rate (``float``): rate of sampling clients
        model_def (``torch.Module``): definition of for constructing the model
        epochs (``int``): number of local epochs
        criterion_def (``Callable``): loss function defining local objective
        optimizer_def (``Callable``): derfintion of server optimizer
        local_optimizer_def (``Callable``): defintoin of local optimizer
        lr_scheduler_def (``Callable``): definition of lr scheduler of server optimizer.
        local_lr_scheduler_def (``Callable``): definition of lr scheduler of local
            optimizer
        r2r_local_lr_scheduler_def (``Callable``): definition to schedule lr that is
            delivered to the clients at each round (deterimined init lr of the
            client optimizer)
        batch_size (int): batch size of the local trianing
        test_batch_size (int): inference time batch size
        device (str): cpu, cuda, or gpu number

    .. note::
        definition of
            * learning rate schedulers, could be any of the ones defined at
                ``torch.optim.lr_scheduler`` or any other that implements step and
                get_last_lr methods._schedulers``.
            * optimizers, could be any ``torch.optim.Optimizer``.
            * model, could be any ``torch.Module``.
            * criterion, could be any ``fedsim.scores.Score``.

    .. _Tackling the Objective Inconsistency Problem in Heterogeneous Federated
        Optimization: https://arxiv.org/abs/2007.07481
    """

    def receive_from_client(
        server_storage,
        client_id,
        client_msg,
        train_split_name,
        serial_aggregator,
        appendix_aggregator,
    ):
        n_train = client_msg["num_samples"][train_split_name]
        weight = n_train / client_msg["num_steps"]

        return serial_aggregation(
            server_storage,
            client_id,
            client_msg,
            train_split_name,
            serial_aggregator,
            train_weight=weight,
        )
