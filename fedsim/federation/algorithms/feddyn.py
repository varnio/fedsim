r""" This file contains an implementation of the following paper:
    Title: "Federated Learning Based on Dynamic Regularization"
    Authors: Durmus Alp Emre Acar, Yue Zhao, Ramon Matas, Matthew Mattina, Paul Whatmough, Venkatesh Saligrama
    Publication date: [28 Sept 2020 (modified: 25 Mar 2021)]
    Link: https://openreview.net/forum?id=B7v4QMR6Z9w
"""
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import accuracy_score
from functools import partial

import torch

from fedsim.federation.algorithms import fedavg
from fedsim.federation.evaluation import local_train_val, inference
from fedsim.federation.utils import (
    vector_to_parameters_like,
    get_metric_scores,
)
from fedsim.utils import apply_on_dict


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

        cloud_params = self.read_server('cloud_params')
        self.write_server('avg_params', cloud_params.detach().clone())
        self.write_server('h', torch.zeros_like(cloud_params))
        for client_id in range(num_clients):
            self.write_client(client_id, 'h', torch.zeros_like(cloud_params))
        # oracle read violation, num_clients read violation
        average_sample = len(self.oracle_dataset['train']) / self.num_clients
        self.write_server('average_sample', average_sample)

    def assign_default_params(self):
        return dict(mu=0.01)

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
            params = parameters_to_vector(model.parameters())
            grad_additive = 0.5 * (params - params_init) - h
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

        # update local h
        pseudo_grads = (
            params_init - \
            parameters_to_vector(model.parameters()).detach().clone().data
            )
        new_h = h + pseudo_grads
        self.write_client(client_id, 'h', new_h)

        # return optimized model parameters and number of train samples
        return dict(
            local_params=parameters_to_vector(model.parameters()),
            num_samples=num_train_samples,
            num_steps=num_steps,
            diverged=diverged,
            train_loss=loss,
            metrics=metrics,
        )

    def receive_from_client(self, client_id, client_msg, aggregation_results):
        weight = 1
        self.agg(client_id, client_msg, aggregation_results, weight=weight)

    def optimize(self, aggr_results):

        # get average gradient
        n_samples = aggr_results.pop('num_samples')
        weight = aggr_results.pop('weight')
        if n_samples > 0:
            param_avg = aggr_results.pop('local_params') / weight
            optimizer = self.read_server('optimizer')
            cloud_params = self.read_server('cloud_params')
            pseudo_grads = cloud_params.data - param_avg
            h = self.read_server('h')
            # read total clients VIOLATION
            h = h + weight / self.num_clients * pseudo_grads
            new_params = param_avg - h

            modified_pseudo_grads = cloud_params.data - new_params
            # update cloud params
            optimizer.zero_grad()
            cloud_params.grad = modified_pseudo_grads
            optimizer.step()
            self.write_server('avg_params', param_avg.detach().clone())
            self.write_server('h', h.data)

            # prepare for report
            n_steps = aggr_results.pop('num_steps')
            normalized_metrics = apply_on_dict(
                aggr_results,
                lambda _, value: value / n_steps,
                return_as_dict=True)
            # purge aggregated results
            del param_avg
        return normalized_metrics

    def report(self, dataloaders, metric_logger, device, optimize_reports):
        # load cloud stuff
        deployment_points = dict(
            cloud=self.read_server('cloud_params'),
            avg=self.read_server('avg_params'),
        )
        model = self.read_server('model')

        for point_name, point in deployment_points.items():
            # copy cloud params to cloud model to send to the client
            vector_to_parameters(point.detach().clone().data,
                                 model.parameters())

            for key, loader in dataloaders.items():
                metrics, _ = inference(
                    model,
                    loader,
                    {'{}.{}_accuracy'.format(point_name, key): accuracy_score},
                    device=device,
                )
                t = self.rounds
                log_fn = metric_logger.add_scalar
                apply_on_dict(metrics, log_fn, global_step=t)
        apply_on_dict(optimize_reports, log_fn, global_step=t)
