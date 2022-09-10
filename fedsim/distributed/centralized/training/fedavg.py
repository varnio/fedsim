r"""
FedAvg
------
"""
import inspect
import math
from functools import partial

from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler

from fedsim.local.training import local_inference
from fedsim.local.training import local_train
from fedsim.local.training.step_closures import default_step_closure
from fedsim.utils import initialize_module
from fedsim.utils import vectorize_module

from ..centralized_fl_algorithm import CentralFLAlgorithm
from .utils import serial_aggregation


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
                ``torch.optim.lr_scheduler`` or any other that implements step and
                get_last_lr methods._schedulers``.
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
        model = self.get_model_def()().to(device)
        params = vectorize_module(model, clone=True, detach=True)
        optimizer = optimizer_def(params=[params])
        lr_scheduler = None
        if lr_scheduler_def is not None:
            lr_scheduler = lr_scheduler_def(optimizer=optimizer)
        # write model and optimizer to server
        server_storage = self.get_server_storage()
        server_storage.write("model", model)
        server_storage.write("cloud_params", params)
        server_storage.write("optimizer", optimizer)
        server_storage.write("lr_scheduler", lr_scheduler)

    def send_to_client(self, server_storage, client_id):
        # since fedavg broadcast the same model to all selected clients,
        # the argument client_id is not used

        # load cloud stuff
        cloud_params = server_storage.read("cloud_params")
        model = server_storage.read("model")
        # copy cloud params to cloud model to send to the client
        initialize_module(model, cloud_params, clone=True, detach=True)
        # return a copy of the cloud model
        return dict(model=model)

    # define client operation
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
        # make the data ready
        train_split_name = train_split_name
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
        if train_split_name in metrics:
            train_scores = metrics[train_split_name]
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
        if rounds % criterion.log_freq == 0:
            metrics_dict[train_split_name][criterion.get_name()] = criterion.get_score()
        num_samples_dict = {train_split_name: num_train_samples}
        # other splits
        for split_name, split in datasets.items():
            if split_name != train_split_name and split_name in metrics:
                o_scores = metrics[split_name]
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
            local_params=vectorize_module(model),
            num_steps=num_steps,
            diverged=diverged,
            num_samples=num_samples_dict,
            metrics=metrics_dict,
        )

    def receive_from_client(
        self,
        server_storage,
        client_id,
        client_msg,
        train_split_name,
        aggregation_results,
    ):
        return serial_aggregation(
            server_storage, client_id, client_msg, train_split_name, aggregation_results
        )

    def optimize(self, server_storage, aggregator):
        if "local_params" in aggregator:
            param_avg = aggregator.pop("local_params")
            optimizer = server_storage.read("optimizer")
            lr_scheduler = server_storage.read("lr_scheduler")
            cloud_params = server_storage.read("cloud_params")
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

    def deploy(self, server_storage):
        return dict(avg=server_storage.read("cloud_params"))

    def report(
        self,
        server_storage,
        dataloaders,
        rounds,
        metrics,
        metric_logger,
        device,
        optimize_reports,
        deployment_points=None,
    ):
        model = server_storage.read("model")
        metrics_from_deployment = dict()
        # TODO: reporting norm and similar metrics should be implemented
        # through hooks (hook probe perhaps)
        norm_report_freq = 50
        norm_reports = dict()
        if deployment_points is not None:
            for point_name, point in deployment_points.items():
                # copy cloud params to cloud model to send to the client
                initialize_module(model, point, clone=True, detach=True)

                for split_name, loader in dataloaders.items():
                    if split_name in metrics:
                        scores = metrics[split_name]
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
                    if rounds % norm_report_freq == 0:
                        norm_reports[
                            f"server.{point_name}.param.norm"
                        ] = point.norm().item()
        return {**metrics_from_deployment, **optimize_reports, **norm_reports}
