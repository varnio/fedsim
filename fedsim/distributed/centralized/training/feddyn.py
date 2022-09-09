r"""
FedDyn
-------
"""
import inspect
from functools import partial

import torch
from torch.optim import SGD

from fedsim.local.training.step_closures import default_step_closure
from fedsim.utils import vector_to_parameters_like
from fedsim.utils import vectorize_module

from . import fedavg
from .utils import serial_aggregation


class FedDyn(fedavg.FedAvg):
    r"""Implements FedDyn algorithm for centralized FL.

    For further details regarding the algorithm we refer to `Federated Learning Based
    on Dynamic Regularization`_.

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
        alpha (float): FedDyn's :math:`\alpha` hyper-parameter for local regularization

    .. note::
        definition of
            * learning rate schedulers, could be any of the ones defined at
                ``torch.optim.lr_scheduler`` or any other that implements step and
                get_last_lr methods._schedulers``.
            * optimizers, could be any ``torch.optim.Optimizer``.
            * model, could be any ``torch.Module``.
            * criterion, could be any ``fedsim.losses``.


    .. _Federated Learning Based on Dynamic Regularization:
        https://openreview.net/forum?id=B7v4QMR6Z9w
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
        alpha=0.1,
    ):

        super(FedDyn, self).__init__(
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
        server_storage.write("alpha", alpha)

        # oracle read violation, num_clients read violation
        print("Warning: private access violation")
        print("\t", end="")
        print(
            "FedDyn assumes prior knowledge on the number of clients and oracle samples"
        )
        private_storage = self.get_server_private_storage()
        oracle_dataset = private_storage.read("oracle_dataset")
        num_clients = private_storage.read("num_clients")
        average_sample = len(oracle_dataset[self.get_train_split_name()]) / num_clients
        server_storage.write("average_sample", average_sample)
        server_storage.write("violation_shouted", False)

    def send_to_client(self, server_storage, client_id):
        msg = super().send_to_client(server_storage, client_id)
        msg["average_sample"] = server_storage.read("average_sample")
        msg["alpha"] = server_storage.read("alpha")
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
        alpha = ctx["alpha"]
        average_sample = ctx["average_sample"]
        params_init = vectorize_module(model, clone=True, detach=True)
        h = storage.read("h")

        alpha_adaptive = alpha / len(datasets[train_split_name]) * average_sample

        def transform_grads_fn(model):
            params = vectorize_module(model)
            grad_additive = 0.5 * (params - params_init)
            if h is not None:
                grad_additive -= h
            grad_additive_list = vector_to_parameters_like(
                alpha_adaptive * grad_additive, model.parameters()
            )

            for p, g_a in zip(model.parameters(), grad_additive_list):
                p.grad += g_a

        step_closure_ = partial(
            default_step_closure, transform_grads=transform_grads_fn
        )
        opt_res = super(FedDyn, self).send_to_server(
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
        new_h = pseudo_grads if h is None else pseudo_grads + h
        storage.write("h", new_h)
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
        return serial_aggregation(
            server_storage,
            client_id,
            client_msg,
            train_split_name,
            aggregation_results,
            train_weight=weight,
        )

    def optimize(self, server_storage, aggregator):
        if "local_params" in aggregator:
            weight = aggregator.get_weight("local_params")
            param_avg = aggregator.pop("local_params")
            optimizer = server_storage.read("optimizer")
            lr_scheduler = server_storage.read("lr_scheduler")
            cloud_params = server_storage.read("cloud_params")
            pseudo_grads = cloud_params.data - param_avg
            h = server_storage.read("h")
            # read total clients VIOLATION
            silent = server_storage.read("violation_shouted")
            private_storage = self.get_server_private_storage(verbose=not silent)
            if not silent:
                server_storage.write("violation_shouted", True)
            num_clients = private_storage.read("num_clients")

            h = h + weight / num_clients * pseudo_grads
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
            server_storage.write("h", h.data)
            # purge aggregated results
            del param_avg
        return aggregator.pop_all()

    def deploy(self, server_storage):
        return dict(
            cloud=server_storage.read("cloud_params"),
            avg=server_storage.read("avg_params"),
        )
