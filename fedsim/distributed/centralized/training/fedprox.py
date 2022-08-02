r"""
FedProx
-------
"""
from functools import partial

from torch.nn.utils import parameters_to_vector

from fedsim.local.training.step_closures import default_step_closure
from fedsim.utils import vector_to_parameters_like

from . import fedavg


class FedProx(fedavg.FedAvg):
    r"""Implements FedProx algorithm for centralized FL.

    For further details regarding the algorithm we refer to `Federated Optimization in
    Heterogeneous Networks`_.

    Args:
        data_manager (Callable): data manager
        metric_logger (Callable): a logger object
        num_clients (int): number of clients
        sample_scheme (str): mode of sampling clients
        sample_rate (float): rate of sampling clients
        model_class (Callable): class for constructing the model
        epochs (int): number of local epochs
        loss_fn (Callable): loss function defining local objective
        batch_size (int): local trianing batch size
        test_batch_size (int): inference time batch size
        local_weight_decay (float): weight decay for local optimization
        slr (float): server learning rate
        clr (float): client learning rate
        clr_decay (float): round to round decay for clr (multiplicative)
        clr_decay_type (str): type of decay for clr (step or cosine)
        min_clr (float): minimum client learning rate
        clr_step_size (int): frequency of applying clr_decay
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
        batch_size=32,
        test_batch_size=64,
        local_weight_decay=0.0,
        slr=1.0,
        clr=0.1,
        clr_decay=1.0,
        clr_decay_type="step",
        min_clr=1e-12,
        clr_step_size=1000,
        device="cuda",
        log_freq=10,
        mu=0.0001,
        *args,
        **kwargs,
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
            batch_size,
            test_batch_size,
            local_weight_decay,
            slr,
            clr,
            clr_decay,
            clr_decay_type,
            min_clr,
            clr_step_size,
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
        lr,
        weight_decay=0,
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
            lr,
            weight_decay,
            device,
            ctx,
            step_closure=step_closure_,
            *args,
            **kwargs,
        )
