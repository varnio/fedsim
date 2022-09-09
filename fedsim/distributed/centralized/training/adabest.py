r"""
AdaBest
-------
"""
import inspect
from functools import partial

import torch
from torch.optim import SGD

from fedsim.local.training.step_closures import default_step_closure
from fedsim.utils import SerialAggregator
from fedsim.utils import vector_to_parameters_like
from fedsim.utils import vectorize_module

from . import fedavg
from .utils import serial_aggregation


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
                ``torch.optim.lr_scheduler`` or any other that implements step and
                get_last_lr methods._schedulers``.
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
        mu=0.1,
        beta=0.8,
    ):
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

        server_storage = self.get_server_storage()
        cloud_params = server_storage.read("cloud_params")
        server_storage.write("avg_params", cloud_params.clone().detach())
        server_storage.write("h", torch.zeros_like(cloud_params))
        server_storage.write("average_sample", 0)
        server_storage.write("mu", mu)
        server_storage.write("beta", beta)

        # for client_id in range(num_clients):
        #     self.write_client(client_id, "h", torch.zeros_like(cloud_params))
        #     self.write_client(client_id, "last_round", -1)

    def send_to_client(self, server_storage, client_id):
        msg = super().send_to_client(server_storage, client_id)
        msg["average_sample"] = server_storage.read("average_sample")
        msg["mu"] = server_storage.read("mu")
        return msg

    def send_to_server(
        self,
        id,
        rounds,
        storage,
        datasets,
        train_split_name,
        metrics,
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
        mu = ctx["mu"]
        average_sample = ctx["average_sample"]
        params_init = vectorize_module(model, clone=True, detach=True)
        h = storage.read("h")
        mu_adaptive = mu / len(datasets[train_split_name]) * average_sample * epochs

        def transform_grads_fn(model):
            if h is not None:
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
            id,
            rounds,
            storage,
            datasets,
            train_split_name,
            metrics,
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
        pseudo_grads = params_init - vectorize_module(model, clone=True, detach=True)
        last_round = storage.read("last_round")
        if h is None:
            new_h = pseudo_grads
        else:
            if last_round is None:
                last_round = rounds - 1
            new_h = 1 / (rounds - last_round) * h + pseudo_grads

        storage.write("h", new_h)
        storage.write("last_round", rounds)
        return opt_res

    def receive_from_client(
        self,
        server_storage,
        client_id,
        client_msg,
        train_split_name,
        aggregation_results,
    ):
        weight = 1
        agg_res = serial_aggregation(
            server_storage,
            client_id,
            client_msg,
            train_split_name,
            aggregation_results,
            train_weight=weight,
        )
        n_train = client_msg["num_samples"][self.get_train_split_name()]
        self.general_agg.add("avg_m", n_train / client_msg["num_steps"], 1)
        server_storage.write("average_sample", self.general_agg.get("avg_m"))
        return agg_res

    def optimize(self, server_storage, aggregator):
        if "local_params" in aggregator:
            beta = server_storage.read("beta")
            param_avg = aggregator.pop("local_params")
            optimizer = server_storage.read("optimizer")
            lr_scheduler = server_storage.read("lr_scheduler")
            cloud_params = server_storage.read("cloud_params")
            h = beta * (server_storage.read("avg_params") - param_avg)
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
            server_storage.write("avg_params", param_avg.detach().clone())
        return aggregator.pop_all()

    def deploy(self, server_storage):
        return dict(
            cloud=server_storage.read("cloud_params"),
            avg=server_storage.read("avg_params"),
        )
