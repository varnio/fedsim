import sys
from copy import deepcopy
from sklearn.metrics import accuracy_score
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from fedsim.federation.base_algorithm import BaseAlgorithm
from fedsim.utils import (
    search_in_submodules,
    add_in_dict,
    add_dict_to_dict,
    apply_on_dict,
)
from fedsim.federation.evaluation import local_train_val, inference


class Algorithm(BaseAlgorithm):

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
        model_class = search_in_submodules('fedsim.models', model)

        # make mode and optimizer
        model = model_class().to(self.device)
        params = deepcopy(
            parameters_to_vector(model.parameters()).clone().detach())
        optimizer = SGD(params=model.parameters(), lr=slr)
        # write model and optimizer to server
        self.write_server('model', model)
        self.write_server('cloud_params', params)
        self.write_server('optimizer', optimizer)

    def assign_default_params(self):
        return None

    def send_to_client(self, client_id):
        # since fedavg broadcast the same model to all selected clients,
        # the argument client_id is not used

        # load cloud stuff
        cloud_params = self.read_server('cloud_params')
        model = self.read_server('model')

        # copy cloud params to cloud model to send to the client
        vector_to_parameters(cloud_params.detach().clone().data,
                             model.parameters())
        # return a copy of the cloud model
        return dict(model=model)

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
                                     })
        num_train_samples, num_steps, diverged, loss, metrics = opt_result
        # return optimized model parameters and number of train samples
        return dict(
            local_params=parameters_to_vector(model.parameters()),
            num_samples=num_train_samples,
            num_steps=num_steps,
            diverged=diverged,
            trian_loss=loss,
            metrics=metrics,
        )

    def agg(self, client_id, client_msg, aggregation_results, weight=1):
        params = client_msg['local_params'].clone().detach().data
        n_samples = client_msg['num_samples']
        n_steps = client_msg['num_steps']
        diverged = client_msg['diverged']
        loss = client_msg['trian_loss']
        metrics = client_msg['metrics']

        if diverged:
            print('client {} diverged'.format(client_id))
            print('exiting ...')
            sys.exit(1)

        add_in_dict('local_params', params, aggregation_results, scale=weight)
        add_in_dict('weight', weight, aggregation_results)
        add_in_dict('num_samples', n_samples, aggregation_results)
        add_in_dict('num_steps', n_steps, aggregation_results)
        add_in_dict('trian_loss', loss, aggregation_results, scale=n_steps)
        add_dict_to_dict(metrics, aggregation_results, scale=n_steps)

        # purge client info
        del client_msg

    def receive_from_client(self, client_id, client_msg, aggregation_results):
        weight = client_msg['num_samples']
        self.agg(client_id, client_msg, aggregation_results, weight=weight)

    def optimize(self, lr, aggr_results):
        # get average gradient
        n_samples = aggr_results.pop('num_samples')
        weight = aggr_results.pop('weight')
        if n_samples > 0:
            param_avg = aggr_results.pop('local_params') / weight

            cloud_params = self.read_server('cloud_params')
            pseudo_grads = cloud_params - param_avg
            # apply sgd
            new_params = cloud_params.data - lr * pseudo_grads.data
            self.write_server('cloud_params', new_params)

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
        deployment_points = dict(avg=self.read_server('cloud_params'), )
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
