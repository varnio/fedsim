r"""
AdaBest
-------
"""
from functools import partial

import torch
from torch.nn.utils import parameters_to_vector

from fedsim.local.training.step_closures import default_step_closure
from fedsim.utils import SerialAggregator
from fedsim.utils import vector_to_parameters_like

from . import fedavg


class AdaBest(fedavg.FedAvg):
    r"""Implements AdaBest algorithm for centralized FL.

    For further details regarding the algorithm we refer to `AdaBest: Minimizing Client
    Drift in Federated Learning via Adaptive Bias Estimation`_.

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
        mu (float): AdaBest's :math:`\mu` parameter for local regularization
        beta (float): AdaBest's :math:`\beta` parameter for global regularization

    .. _AdaBest\: Minimizing Client Drift in Federated Learning via Adaptive
        Bias Estimation: https://arxiv.org/abs/2204.13170
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
        mu=0.02,
        beta=0.98,
        *args,
        **kwargs,
    ):
        self.mu = mu
        self.beta = beta
        # this is to avoid violations like reading oracle info and
        # number of clients in FedDyn and SCAFFOLD
        self.general_agg = SerialAggregator()

        super(AdaBest, self).__init__(
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

        cloud_params = self.read_server("cloud_params")
        self.write_server("avg_params", cloud_params.detach().clone())

        for client_id in range(num_clients):
            self.write_client(client_id, "h", torch.zeros_like(cloud_params))
            self.write_client(client_id, "last_round", -1)
        self.write_server("average_sample", 0)

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
        h = self.read_client(client_id, "h")
        mu_adaptive = (
            self.mu / len(datasets["train"]) * self.read_server("average_sample")
        )

        def transform_grads_fn(model):
            grad_additive = -h
            grad_additive_list = vector_to_parameters_like(
                mu_adaptive * grad_additive, model.parameters()
            )

            for p, g_a in zip(model.parameters(), grad_additive_list):
                p.grad += g_a

        step_closure_ = partial(
            default_step_closure, transform_grads=transform_grads_fn
        )
        opt_res = super(AdaBest, self).send_to_server(
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

        # update local h
        pseudo_grads = (
            params_init - parameters_to_vector(model.parameters()).detach().clone().data
        )
        t = self.rounds
        new_h = 1 / (t - self.read_client(client_id, "last_round")) * h + pseudo_grads
        self.write_client(client_id, "h", new_h)
        self.write_client(client_id, "last_round", self.rounds)
        return opt_res

    def receive_from_client(self, client_id, client_msg, aggregation_results):
        weight = 1
        self.agg(client_id, client_msg, aggregation_results, weight=weight)
        self.general_agg.add("avg_m", client_msg["num_samples"] / self.epochs, 1)
        self.write_server("average_sample", self.general_agg.get("avg_m"))

    def optimize(self, aggregator):
        if "local_params" in aggregator:
            param_avg = aggregator.pop("local_params")
            optimizer = self.read_server("optimizer")
            cloud_params = self.read_server("cloud_params")
            h = self.beta * (self.read_server("avg_params") - param_avg)
            new_params = param_avg - h
            modified_pseudo_grads = cloud_params.data - new_params
            # update cloud params
            optimizer.zero_grad()
            cloud_params.grad = modified_pseudo_grads
            optimizer.step()
            self.write_server("avg_params", param_avg.detach().clone())
        return aggregator.pop_all()

    def deploy(self):
        return dict(
            cloud=self.read_server("cloud_params"),
            avg=self.read_server("avg_params"),
        )
