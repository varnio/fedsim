r"""
FedAvg
------
"""
import inspect
import math
import sys
from copy import deepcopy
from functools import partial

from torch.nn.utils import parameters_to_vector
from torch.nn.utils import vector_to_parameters
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler

from fedsim.local.training import local_inference
from fedsim.local.training import local_train
from fedsim.local.training.step_closures import default_step_closure

from ..centralized_fl_algorithm import CentralFLAlgorithm


class FedAvg(CentralFLAlgorithm):
    r"""Implements FedAvg algorithm for centralized FL.

    For further details regarding the algorithm we refer to `Communication-Efficient
    Learning of Deep Networks from Decentralized Data`_.

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

    .. note::
        definition of
        * learning rate schedulers, could be any of the ones defined at
        ``fedsim.lr_schedulers``.
        * optimizers, could be any ``torch.optim.Optimizer``.
        * model, could be any ``torch.Module``.
        * criterion, could be any ``fedsim.losses``.

    .. _Communication-Efficient Learning of Deep Networks from Decentralized
        Data: https://arxiv.org/abs/1602.05629
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
    ):
        super(FedAvg, self).__init__(
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

        # make mode and optimizer
        model = self.get_model_def()().to(self.device)
        params = deepcopy(parameters_to_vector(model.parameters()).clone().detach())
        optimizer = optimizer_def(params=[params])
        lr_scheduler = None
        if lr_scheduler_def is not None:
            lr_scheduler = lr_scheduler_def(optimizer=optimizer)
        # write model and optimizer to server
        self.write_server("model", model)
        self.write_server("cloud_params", params)
        self.write_server("optimizer", optimizer)
        self.write_server("lr_scheduler", lr_scheduler)

    def send_to_client(self, client_id):
        # since fedavg broadcast the same model to all selected clients,
        # the argument client_id is not used

        # load cloud stuff
        cloud_params = self.read_server("cloud_params")
        model = self.read_server("model")

        # copy cloud params to cloud model to send to the client
        vector_to_parameters(cloud_params.detach().clone().data, model.parameters())
        # return a copy of the cloud model
        return dict(model=model)

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
        # create a random sampler with replacement so that
        # stochasticity is maximiazed and privacy is not compromized
        sampler = RandomSampler(
            datasets[train_split_name],
            replacement=True,
            num_samples=math.ceil(len(datasets[train_split_name]) / train_batch_size)
            * train_batch_size,
        )
        # # create train data loader
        train_loader = DataLoader(
            datasets[train_split_name], batch_size=train_batch_size, sampler=sampler
        )

        model = ctx["model"]
        optimizer = optimizer_def(model.parameters())

        if lr_scheduler_def is not None:
            lr_scheduler = lr_scheduler_def(optimizer=optimizer)
        else:
            lr_scheduler = None
        # optimize the model locally
        step_closure_ = default_step_closure if step_closure is None else step_closure
        if train_split_name in round_scores:
            train_scores = round_scores[train_split_name]
        else:
            train_scores = dict()
        num_train_samples, num_steps, diverged, = local_train(
            model,
            train_loader,
            epochs,
            0,
            criterion,
            optimizer,
            lr_scheduler,
            device,
            step_closure_,
            scores=train_scores,
        )
        # get average train scores
        metrics_dict = {
            train_split_name: {
                name: score.get_score() for name, score in train_scores.items()
            }
        }
        # append train loss
        if self.rounds % criterion.log_freq == 0:
            metrics_dict[train_split_name][criterion.get_name()] = criterion.get_score()
        num_samples_dict = {train_split_name: num_train_samples}
        # other splits
        for split_name, split in datasets.items():
            if split_name != train_split_name and split_name in round_scores:
                o_scores = round_scores[split_name]
                split_loader = DataLoader(
                    split,
                    batch_size=inference_batch_size,
                    shuffle=False,
                )
                num_samples = local_inference(
                    model,
                    split_loader,
                    scores=o_scores,
                    device=device,
                )
                metrics_dict[split_name] = {
                    name: score.get_score() for name, score in o_scores.items()
                }
                num_samples_dict[split_name] = num_samples
        # return optimized model parameters and number of train samples
        return dict(
            local_params=parameters_to_vector(model.parameters()),
            num_steps=num_steps,
            diverged=diverged,
            num_samples=num_samples_dict,
            metrics=metrics_dict,
        )

    def agg(
        self,
        client_id,
        client_msg,
        aggregator,
        train_weight=None,
        other_weight=None,
    ):
        params = client_msg["local_params"].clone().detach().data
        diverged = client_msg["diverged"]
        metrics = client_msg["metrics"]
        n_samples = client_msg["num_samples"]

        if diverged:
            print("client {} diverged".format(client_id))
            print("exiting ...")
            sys.exit(1)
        if train_weight is None:
            train_weight = n_samples[self.get_train_split_name()]

        if train_weight > 0:
            aggregator.add("local_params", params, train_weight)
            for split_name, metrics in metrics.items():
                if other_weight is None:
                    other_weight = n_samples[split_name]
                for key, metric in metrics.items():
                    aggregator.add(f"clients.{split_name}.{key}", metric, other_weight)

        # purge client info
        del client_msg

    def receive_from_client(self, client_id, client_msg, aggregation_results):
        self.agg(client_id, client_msg, aggregation_results)

    def optimize(self, aggregator):
        if "local_params" in aggregator:
            param_avg = aggregator.pop("local_params")
            optimizer = self.read_server("optimizer")
            lr_scheduler = self.read_server("lr_scheduler")
            cloud_params = self.read_server("cloud_params")
            pseudo_grads = cloud_params.data - param_avg
            # update cloud params
            optimizer.zero_grad()
            cloud_params.grad = pseudo_grads
            optimizer.step()
            if lr_scheduler is not None:
                step_args = inspect.signature(lr_scheduler.step).parameters
                if "metrics" in step_args:
                    trigger_metric = lr_scheduler.trigger_metric
                    lr_scheduler.step(aggregator.get(trigger_metric))
                else:
                    lr_scheduler.step()
            # purge aggregated results
            del param_avg
        return aggregator.pop_all()

    def deploy(self):
        return dict(avg=self.read_server("cloud_params"))

    def report(
        self,
        dataloaders,
        round_scores,
        metric_logger,
        device,
        optimize_reports,
        deployment_points=None,
    ):
        model = self.read_server("model")
        metrics_from_deployment = dict()
        if deployment_points is not None:
            for point_name, point in deployment_points.items():
                # copy cloud params to cloud model to send to the client
                vector_to_parameters(point.detach().clone().data, model.parameters())

                for split_name, loader in dataloaders.items():
                    if split_name in round_scores:
                        scores = round_scores[split_name]
                        _ = local_inference(
                            model,
                            loader,
                            scores=scores,
                            device=device,
                        )
                        split_metrics = {
                            f"server.{point_name}.{split_name}."
                            f"{score_name}": score.get_score()
                            for score_name, score in scores.items()
                        }
                        metrics_from_deployment = {
                            **metrics_from_deployment,
                            **split_metrics,
                        }
        return {**metrics_from_deployment, **optimize_reports}
