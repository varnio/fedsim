r""" This file contains an implementation of the following paper:
    Title: "Federated Learning Based on Dynamic Regularization"
    Authors: Durmus Alp Emre Acar, Yue Zhao, Ramon Matas, Matthew Mattina,
    ----- Paul Whatmough, Venkatesh Saligrama
    Publication date: [28 Sept 2020 (modified: 25 Mar 2021)]
    Link: https://openreview.net/forum?id=B7v4QMR6Z9w
"""
from functools import partial

import torch
from torch.nn.utils import parameters_to_vector

from fedsim.local.training.step_closures import default_closure
from fedsim.utils import vector_to_parameters_like

from . import fedavg


class FedDyn(fedavg.FedAvg):
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
        *args,
        **kwargs,
    ):
        self.mu = mu

        super(FedDyn, self).__init__(
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
        self.write_server("h", torch.zeros_like(cloud_params))
        for client_id in range(num_clients):
            self.write_client(client_id, "h", torch.zeros_like(cloud_params))
        # oracle read violation, num_clients read violation
        average_sample = len(self.oracle_dataset["train"]) / self.num_clients
        self.write_server("average_sample", average_sample)

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
            self.mu
            / len(datasets["train"])
            * self.read_server("average_sample")
        )

        def transform_grads_fn(model):
            params = parameters_to_vector(model.parameters())
            grad_additive = 0.5 * (params - params_init) - h
            grad_additive_list = vector_to_parameters_like(
                mu_adaptive * grad_additive, model.parameters()
            )

            for p, g_a in zip(model.parameters(), grad_additive_list):
                p.grad += g_a

        step_closure_ = partial(
            default_closure, transform_grads=transform_grads_fn
        )
        opt_res = super(FedDyn, self).send_to_server(
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
            params_init
            - parameters_to_vector(model.parameters()).detach().clone().data
        )
        new_h = h + pseudo_grads
        self.write_client(client_id, "h", new_h)
        return opt_res

    def receive_from_client(self, client_id, client_msg, aggregation_results):
        weight = 1
        self.agg(client_id, client_msg, aggregation_results, weight=weight)

    def optimize(self, aggregator):
        if "local_params" in aggregator:
            weight = aggregator.get_weight("local_params")
            param_avg = aggregator.pop("local_params")
            optimizer = self.read_server("optimizer")
            cloud_params = self.read_server("cloud_params")
            pseudo_grads = cloud_params.data - param_avg
            h = self.read_server("h")
            # read total clients VIOLATION
            h = h + weight / self.num_clients * pseudo_grads
            new_params = param_avg - h
            modified_pseudo_grads = cloud_params.data - new_params
            # update cloud params
            optimizer.zero_grad()
            cloud_params.grad = modified_pseudo_grads
            optimizer.step()
            self.write_server("avg_params", param_avg.detach().clone())
            self.write_server("h", h.data)
            # purge aggregated results
            del param_avg
        return aggregator.pop_all()

    def deploy(self):
        return dict(
            cloud=self.read_server("cloud_params"),
            avg=self.read_server("avg_params"),
        )
