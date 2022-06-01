from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import accuracy_score
from functools import partial

import torch

from federation.algorithms.feddyn import Algorithm
from federation.evaluation import local_train_val, inference
from federation.utils import vector_to_parameters_like, get_metric_scores
from utils import apply_on_dict


class Algorithm(Algorithm):

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
            trian_loss=loss,
            metrics=metrics,
        )

    def optimize(self, lr, aggr_results):

        # get average gradient
        n_samples = aggr_results.pop('num_samples')
        if n_samples > 0:
            counter = n_samples = aggr_results.pop('counter')
            param_avg = aggr_results.pop('local_params') / counter

            cloud_params = self.read_server('cloud_params')
            # read total clients violation
            h = self.beta * (self.read_server('avg_params') - param_avg)
            new_params = param_avg - h

            modified_pseudo_grads = cloud_params - new_params
            # apply sgd
            new_params = cloud_params.data - lr * modified_pseudo_grads.data
            self.write_server('cloud_params', new_params)
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
