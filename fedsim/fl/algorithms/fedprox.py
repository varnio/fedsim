r""" This file contains an implementation of the following paper:
    Title: "Federated Optimization in Heterogeneous Networks"
    Authors: Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, Virginia Smith
    Publication date: [Submitted on 14 Dec 2018 (v1), last revised 21 Apr 2020 (this version, v5)]
    Link: https://arxiv.org/abs/1812.06127
"""
from torch.nn.utils import parameters_to_vector
from functools import partial

from ..utils import default_closure, vector_to_parameters_like

from . import fedavg


class FedProx(fedavg.FedAvg):

    def __init__(
        self,
        data_manager,
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
        metric_logger,
        device,
        log_freq,
        mu=0.0001,
        *args,
        **kwargs,
    ):
        self.mu = mu

        super(FedProx, self).__init__(
            data_manager,
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
            metric_logger,
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
        device='cuda',
        ctx=None,
        *args,
        **kwargs,
    ):
        model = ctx['model']
        params_init = parameters_to_vector(model.parameters()).detach().clone()
        mu = self.mu

        def transform_grads_fn(model):
            params = parameters_to_vector(model.parameters())
            grad_additive = 0.5 * (params - params_init)
            grad_additive_list = vector_to_parameters_like(
                mu * grad_additive, model.parameters())

            for p, g_a in zip(model.parameters(), grad_additive_list):
                p.grad += g_a

        step_closure_ = partial(default_closure,
                                transform_grads=transform_grads_fn)
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
