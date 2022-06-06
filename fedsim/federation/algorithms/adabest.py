r""" This file contains an implementation of the following paper:
    Title: "Minimizing Client Drift in Federated Learning via Adaptive Bias Estimation"
    Authors: Farshid Varno, Marzie Saghayi, Laya Rafiee, Sharut Gupta, Stan Matwin, Mohammad Havaei
    Publication date: [Submitted on 27 Apr 2022 (v1), last revised 23 May 2022 (this version, v2)]
    Link: https://arxiv.org/abs/2204.13170
"""
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import accuracy_score
from functools import partial

import torch

from fedsim.federation.algorithms import feddyn
from fedsim.federation.evaluation import local_train_val
from fedsim.federation.utils import vector_to_parameters_like, get_metric_scores
from fedsim.utils import apply_on_dict


# TODO: add dynamic avg_m to avoid violation of reading prior info
class AdaBest(feddyn.FedDyn):

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
        metric_logger,
        device,
        log_freq,
        mu=0.02,
        beta=0.98,
        *args,
        **kwargs,
    ):
        self.beta = beta

        super(AdaBest, self).__init__(
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
            metric_logger,
            device,
            log_freq,
            mu=mu,
        )

        cloud_params = self.read_server('cloud_params')
        self.write_server('avg_params', cloud_params.detach().clone())

        for client_id in range(num_clients):
            self.write_client(client_id, 'h', torch.zeros_like(cloud_params))
            self.write_client(client_id, 'last_round', -1)
        # oracle read violation, num_clients read violation
        average_sample = len(self.oracle_dataset['train']) / self.num_clients
        self.write_server('average_sample', average_sample)

    def assign_default_params(self):
        return dict(mu=0.01, beta=0.9)

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
        train_loader = DataLoader(
            datasets[data_split_name],
            batch_size=batch_size,
            shuffle=False,
        )
        model = ctx['model']
        optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

        params_init = parameters_to_vector(model.parameters()).detach().clone()
        h = self.read_client(client_id, 'h')
        mu_adaptive = self.mu / len(datasets['train']) *\
            self.read_server('average_sample')

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
            grad_additive = -h
            grad_additive_list = vector_to_parameters_like(
                mu_adaptive * grad_additive, model.parameters())

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
                                     train_loader,
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

        # update local h
        pseudo_grads = (
            params_init - \
            parameters_to_vector(model.parameters()).detach().clone().data
            )
        t = self.rounds
        new_h = 1 / (t - self.read_client(client_id, 'last_round')) * h +\
            pseudo_grads
        self.write_client(client_id, 'h', new_h)
        self.write_client(client_id, 'last_round', self.rounds)

        # return optimized model parameters and number of train samples
        return dict(
            local_params=parameters_to_vector(model.parameters()),
            num_samples=num_train_samples,
            num_steps=num_steps,
            diverged=diverged,
            train_loss=loss,
            metrics=metrics,
        )

    def optimize(self, aggr_results):

        # get average gradient
        n_samples = aggr_results.pop('num_samples')
        weight = aggr_results.pop('weight')
        if n_samples > 0:
            param_avg = aggr_results.pop('local_params') / weight
            optimizer = self.read_server('optimizer')
            cloud_params = self.read_server('cloud_params')
            # read total clients violation
            h = self.beta * (self.read_server('avg_params') - param_avg)
            new_params = param_avg - h

            modified_pseudo_grads = cloud_params.data - new_params
            # update cloud params
            optimizer.zero_grad()
            cloud_params.grad = modified_pseudo_grads
            optimizer.step()
            self.write_server('avg_params', param_avg.detach().clone())

            # prepare for report
            n_steps = aggr_results.pop('num_steps')
            normalized_metrics = apply_on_dict(
                aggr_results,
                lambda _, value: value / n_steps,
                return_as_dict=True)
            # purge aggregated results
            del param_avg
        return normalized_metrics
