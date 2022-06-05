r""" This file contains an implementation of the following paper:
    Title: "Federated Optimization in Heterogeneous Networks"
    Authors: Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, Virginia Smith
    Publication date: [Submitted on 14 Dec 2018 (v1), last revised 21 Apr 2020 (this version, v5)]
    Link: https://arxiv.org/abs/1812.06127
"""
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from fedsim.federation.evaluation import local_train_val
from torch.nn.utils import parameters_to_vector
from torch.optim import SGD
from functools import partial
from torch.nn.utils import clip_grad_norm_
import torch

from fedsim.federation.utils import (
    vector_to_parameters_like,
    get_metric_scores,
)

from fedsim.federation.algorithms import fedavg


class Algorithm(fedavg.Algorithm):

    def __init__(
        self,
        data_manager,
        num_clients,
        sample_scheme,
        sample_rate,
        model,
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
        algorithm_params,
        metric_logger,
        device,
        log_freq,
        verbosity,
    ):
        super(Algorithm, self).__init__(
            data_manager,
            num_clients,
            sample_scheme,
            sample_rate,
            model,
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
            algorithm_params,
            metric_logger,
            device,
            log_freq,
            verbosity,
        )

    def assign_default_params(self):
        return dict(mu=0.0001)

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
    ):
        data_split_name = 'train'
        # create train data loader
        train_laoder = DataLoader(
            datasets[data_split_name],
            batch_size=batch_size,
            shuffle=False,
        )
        model = ctx['model']
        optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

        params_init = parameters_to_vector(model.parameters()).detach().clone()
        mu = self.mu

        # closure to be performed at each local step
        def step_closure(x,
                         y,
                         model,
                         loss_fn,
                         optimizer,
                         metric_fn_dict,
                         max_grad_norm=1000,
                         link_fn=partial(torch.argmax, dim=1),
                         device='cpu',
                         **kwargs):
            y_true = y.tolist()
            x = x.to(device)
            y = y.reshape(-1).long()
            y = y.to(device)
            model.train()
            outputs = model(x)
            loss = loss_fn(outputs, y)
            if loss.isnan() or loss.isinf():
                return loss
            # backpropagation
            loss.backward()
            params = parameters_to_vector(model.parameters())
            grad_additive = 0.5 * (params - params_init)
            grad_additive_list = vector_to_parameters_like(
                mu * grad_additive, model.parameters())

            for p, g_a in zip(model.parameters(), grad_additive_list):
                p.grad += g_a
            # Clip gradients
            clip_grad_norm_(parameters=model.parameters(),
                            max_norm=max_grad_norm)
            # optimize
            optimizer.step()
            optimizer.zero_grad()
            y_pred = link_fn(outputs).tolist()
            metrics = get_metric_scores(metric_fn_dict, y_true, y_pred)
            return loss, metrics

        # optimize the model locally
        opt_result = local_train_val(model,
                                     train_laoder,
                                     epochs,
                                     0,
                                     loss_fn,
                                     optimizer,
                                     device,
                                     metric_fn_dict={
                                         '{}_accuracy'.format(data_split_name):
                                         accuracy_score,
                                     },
                                     step_closure=step_closure)
        num_train_samples, num_steps, diverged, loss, metrics = opt_result
        # return optimized model parameters and number of train samples
        return dict(
            local_params=parameters_to_vector(model.parameters()),
            num_samples=num_train_samples,
            num_steps=num_steps,
            diverged=diverged,
            train_loss=loss,
            metrics=metrics,
        )
