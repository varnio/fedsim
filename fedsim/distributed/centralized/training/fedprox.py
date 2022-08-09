r"""
FedProx
-------
"""
from functools import partial

from torch.nn.utils import parameters_to_vector
from torch.optim import SGD

from fedsim.local.training.step_closures import default_step_closure
from fedsim.utils import vector_to_parameters_like

from . import fedavg


class FedProx(fedavg.FedAvg):
    r"""Implements FedProx algorithm for centralized FL.

    For further details regarding the algorithm we refer to `Federated Optimization in
    Heterogeneous Networks`_.

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
        mu (float): FedProx's :math:`\mu` parameter for local regularization

    .. _Federated Optimization in Heterogeneous Networks:
        https://arxiv.org/abs/1812.06127
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
        mu=0.0001,
    ):
        self.mu = mu

        super(FedProx, self).__init__(
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

    def send_to_server(
        self,
        client_id,
        datasets,
        epochs,
        loss_fn,
        batch_size,
        optimizer_class,
        lr_scheduler_class=None,
        device="cuda",
        ctx=None,
        *args,
        **kwargs,
    ):
        model = ctx["model"]
        params_init = parameters_to_vector(model.parameters()).detach().clone()
        mu = self.mu

        def transform_grads_fn(model):
            params = parameters_to_vector(model.parameters())
            grad_additive = 0.5 * (params - params_init)
            grad_additive_list = vector_to_parameters_like(
                mu * grad_additive, model.parameters()
            )

            for p, g_a in zip(model.parameters(), grad_additive_list):
                p.grad += g_a

        step_closure_ = partial(
            default_step_closure, transform_grads=transform_grads_fn
        )
        return super(FedProx, self).send_to_server(
            client_id,
            datasets,
            epochs,
            loss_fn,
            batch_size,
            optimizer_class,
            lr_scheduler_class,
            device,
            ctx,
            step_closure=step_closure_,
            *args,
            **kwargs,
        )
