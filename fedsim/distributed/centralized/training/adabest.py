r"""
AdaBest
-------
"""
import inspect
from functools import partial

import torch
from torch.nn.utils import parameters_to_vector
from torch.optim import SGD

from fedsim.local.training.step_closures import default_step_closure
from fedsim.utils import SerialAggregator
from fedsim.utils import vector_to_parameters_like

from . import fedavg


class AdaBest(fedavg.FedAvg):
    r"""Implements AdaBest algorithm for centralized FL.

    For further details regarding the algorithm we refer to `AdaBest: Minimizing Client
    Drift in Federated Learning via Adaptive Bias Estimation`_.

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
        mu (float): AdaBest's :math:`\mu` hyper-parameter for local regularization
        beta (float): AdaBest's :math:`\beta` hyper-parameter for global regularization

    .. note::
        definition of
        * learning rate schedulers, could be any of the ones defined at
        ``fedsim.lr_schedulers``.
        * optimizers, could be any ``torch.optim.Optimizer``.
        * model, could be any ``torch.Module``.
        * criterion, could be any ``fedsim.losses``.


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
        mu=0.02,
        beta=0.98,
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
        round_scores,
        epochs,
        criterion,
        train_batch_size,
        inference_batch_size,
        optimizer_def,
        lr_scheduler_def=None,
        device="cuda",
        ctx=None,
        step_closure=None,
    ):
        train_split_name = self.get_train_split_name()
        model = ctx["model"]
        params_init = parameters_to_vector(model.parameters()).detach().clone()
        h = self.read_client(client_id, "h")
        mu_adaptive = (
            self.mu
            / len(datasets[train_split_name])
            * self.read_server("average_sample")
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
            round_scores,
            epochs,
            criterion,
            train_batch_size,
            inference_batch_size,
            optimizer_def,
            lr_scheduler_def,
            device,
            ctx,
            step_closure=step_closure_,
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
        self.agg(client_id, client_msg, aggregation_results, train_weight=weight)
        n_train = client_msg["num_samples"][self.get_train_split_name()]
        self.general_agg.add("avg_m", n_train / self.epochs, 1)
        self.write_server("average_sample", self.general_agg.get("avg_m"))

    def optimize(self, aggregator):
        if "local_params" in aggregator:
            param_avg = aggregator.pop("local_params")
            optimizer = self.read_server("optimizer")
            lr_scheduler = self.read_server("lr_scheduler")
            cloud_params = self.read_server("cloud_params")
            h = self.beta * (self.read_server("avg_params") - param_avg)
            new_params = param_avg - h
            modified_pseudo_grads = cloud_params.data - new_params
            # update cloud params
            optimizer.zero_grad()
            cloud_params.grad = modified_pseudo_grads
            optimizer.step()
            if lr_scheduler is not None:
                step_args = inspect.signature(lr_scheduler.step).parameters
                if "metrics" in step_args:
                    trigger_metric = lr_scheduler.trigger_metric
                    lr_scheduler.step(aggregator.get(trigger_metric))
                else:
                    lr_scheduler.step()
            self.write_server("avg_params", param_avg.detach().clone())
        return aggregator.pop_all()

    def deploy(self):
        return dict(
            cloud=self.read_server("cloud_params"),
            avg=self.read_server("avg_params"),
        )
