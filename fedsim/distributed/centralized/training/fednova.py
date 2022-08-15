r"""
FedNova
-------
"""
from functools import partial

from torch.optim import SGD

from . import fedavg


class FedNova(fedavg.FedAvg):
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
        ``fedsim.lr_schedulers``.
        * optimizers, could be any ``torch.optim.Optimizer``.
        * model, could be any ``torch.Module``.
        * criterion, could be any ``fedsim.losses``.
    .. _Tackling the Objective Inconsistency Problem in Heterogeneous Federated
        Optimization: https://arxiv.org/abs/2007.07481
    """

    def __init__(
        self,
        data_manager,
        metric_logger,
        num_clients,
        sample_scheme,
        sample_rate,
        model_def,
        epochs,
        criterion_def,
        optimizer_def=partial(SGD, lr=0.1, weight_decay=0.001),
        local_optimizer_def=partial(SGD, lr=1.0),
        lr_scheduler_def=None,
        local_lr_scheduler_def=None,
        r2r_local_lr_scheduler_def=None,
        batch_size=32,
        test_batch_size=64,
        device="cuda",
    ):
        super(FedNova, self).__init__(
            data_manager,
            metric_logger,
            num_clients,
            sample_scheme,
            sample_rate,
            model_def,
            epochs,
            criterion_def,
            optimizer_def,
            local_optimizer_def,
            lr_scheduler_def,
            local_lr_scheduler_def,
            r2r_local_lr_scheduler_def,
            batch_size,
            test_batch_size,
            device,
        )

    def receive_from_client(self, client_id, client_msg, aggregation_results):
        train_split_name = self.get_train_split_name()
        n_train = client_msg["num_samples"][train_split_name]
        weight = n_train / client_msg["num_steps"]
        self.agg(client_id, client_msg, aggregation_results, train_weight=weight)
