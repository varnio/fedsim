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
        data_manager (Callable): data manager
        metric_logger (Callable): a logall.Logger instance
        num_clients (int): number of clients
        sample_scheme (str): mode of sampling clients
        sample_rate (float): rate of sampling clients
        model_class (Callable): class for constructing the model
        epochs (int): number of local epochs
        loss_fn (Callable): loss function defining local objective
        optimizer_class (Callable): server optimizer class
        local_optimizer_class (Callable): local optimization class
        lr_scheduler_class: class definition for lr scheduler of server optimizer
        local_lr_scheduler_class: class definition for lr scheduler of local optimizer
        r2r_local_lr_scheduler_class: class definition to schedule lr delivered to
            clients at each round (init lr of the client optimizer)
        batch_size (int): local trianing batch size
        test_batch_size (int): inference time batch size
        device (str): cpu, cuda, or gpu number
        log_freq (int): frequency of logging

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
        model_class,
        epochs,
        loss_fn,
        optimizer_class=partial(SGD, lr=0.1, weight_decay=0.001),
        local_optimizer_class=partial(SGD, lr=1.0),
        lr_scheduler_class=None,
        local_lr_scheduler_class=None,
        r2r_local_lr_scheduler_class=None,
        batch_size=32,
        test_batch_size=64,
        device="cuda",
        log_freq=10,
    ):
        super(FedNova, self).__init__(
            data_manager,
            metric_logger,
            num_clients,
            sample_scheme,
            sample_rate,
            model_class,
            epochs,
            loss_fn,
            optimizer_class,
            local_optimizer_class,
            lr_scheduler_class,
            local_lr_scheduler_class,
            r2r_local_lr_scheduler_class,
            batch_size,
            test_batch_size,
            device,
            log_freq,
        )

    def receive_from_client(self, client_id, client_msg, aggregation_results):
        weight = client_msg["num_samples"] / client_msg["num_steps"]
        self.agg(client_id, client_msg, aggregation_results, weight=weight)
